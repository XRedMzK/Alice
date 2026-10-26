"""Strict single-token acquisition boundary for the full-rank payload version."""

from .runtime import MemoryRuntime


class PayloadMemoryRuntime(MemoryRuntime):
    def __init__(self, loaded, bridge, memory, max_tokens=128):
        super().__init__(loaded, bridge, memory, max_tokens)
        if (getattr(bridge, "payload_shape", None) != (17, 128)
                or memory.matrix.shape != (len(memory.source_ids), 17, 128, 128)):
            raise ValueError("Payload runtime requires the versioned 17-head memory boundary")

    def _value_feature(self, text):
        ids = self._ids(" " + text)
        if ids.shape[1] != 1:
            raise ValueError("Payload v1 supports one-token values; multi-token values are not silently averaged")
        return self._prefix(ids).float()[0, 0]

    def answer_with_context(self, request, history_segments, current_turn_ids, *,
                            history_message_count: int, mode="on", max_new_tokens=8):
        """Answer with a bounded active-session suffix and optional neural read.

        ``history_segments`` is lazy, newest first, with ``(source_id, tuple(ids))``
        items. Only a contiguous suffix is retained; the first non-fitting item
        ends iteration. The caller supplies the total count without encoding old
        messages. Neither this method nor its key path reads a transcript file.
        ``current_turn_ids`` already contains the caller's current role message
        and Agent prefix. The legacy question-only answer method is unchanged.
        """
        import time
        import torch

        from alice_runtime.generation import cut_turn
        from .runtime import Question

        if (not isinstance(request, Question) or not isinstance(request.text, str)
                or not request.text.strip()):
            raise ValueError("A nonempty current Question is required")
        if mode not in {"on", "off", "empty"}:
            raise ValueError("Conversation memory mode must be on, off or empty")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise ValueError("Conversation output budget must be 1..64 tokens")
        if (type(history_message_count) is not int
                or not 0 <= history_message_count <= (1 << 63) - 1):
            raise ValueError("History count must be a bounded nonnegative integer")
        if (not isinstance(current_turn_ids, tuple) or not current_turn_ids
                or len(current_turn_ids) + max_new_tokens > self.max_tokens):
            raise ValueError("Current turn plus reserved output exceeds the context budget")
        vocabulary = self.model.get_input_embeddings().weight.shape[0]

        def token_tensor(ids):
            if any(type(token) is not int or not 0 <= token < vocabulary for token in ids):
                raise ValueError("Context token IDs must be valid integer vocabulary indices")
            return torch.tensor([ids], device=self.device, dtype=torch.long)

        current = token_tensor(current_turn_ids)
        started = time.perf_counter()
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        self.context(request)  # Validate both source IDs and the current question.
        before = self.memory.content_digest()
        confidence = readout_norm = 0.0
        applied = False
        with torch.no_grad():
            if mode == "on":
                query = self.key_for(request.text)  # History and answer text never enter this key.
                readout = self.memory.matrix[self.memory.source_index(request.knowledge_source_id)] @ query
                confidence, readout_norm = float(readout[0, 0]), float(readout.norm())
                applied = self.bridge.accepts(readout)
            # Preserve the existing inner repeated-QA payload layout exactly.
            # OFF/EMPTY and rejected reads use only ordinary role/context input.
            base = (self._lexical_input(request.text, readout)
                    if applied else self._prefix(current))
            remaining = self.max_tokens - base.shape[1] - max_new_tokens
            if remaining < 0:
                raise ValueError("Memory layout plus reserved output exceeds the context budget")
            retained, seen = [], 0
            if remaining and history_message_count:
                exhausted = True
                for item in history_segments:
                    seen += 1
                    if seen > history_message_count:
                        raise ValueError("History iterator exceeds its declared message count")
                    if (not isinstance(item, tuple) or len(item) != 2
                            or not isinstance(item[0], str) or not item[0].strip() or len(item[0]) > 96
                            or not isinstance(item[1], tuple) or not item[1]):
                        raise ValueError("History requires source IDs and nonempty token tuples")
                    source, ids = item
                    if len(ids) > remaining:
                        exhausted = False
                        break  # Never skip a newer message to include older context.
                    retained.append((source, token_tensor(ids)))
                    remaining -= len(ids)
                    if not remaining:
                        exhausted = False
                        break  # Do not encode another message when no room remains.
                if exhausted and seen != history_message_count:
                    raise ValueError("History iterator ended before its declared message count")
            chronological = list(reversed(retained))
            # Budget eviction must not leave a reply detached from its prior
            # incoming turn. This generic boundary ignores memory relevance and
            # never removes the separately supplied current turn/base.
            if len(retained) < history_message_count:
                while chronological and chronological[0][0] == "agent":
                    chronological.pop(0)
            embeddings = torch.cat([*(self._prefix(ids) for _, ids in chronological), base], dim=1)
            input_tokens = embeddings.shape[1]
            generated, finish_reason = [], "token_limit"
            boundary = None
            eos = self.model.config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else (eos,))
            for _ in range(max_new_tokens):
                logits = self._forward(embeddings, last_only=True).logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite Qwen output")
                token = int(logits.argmax())
                if token in eos_ids:
                    finish_reason = "eos"
                    break
                generated.append(token)
                raw = self.tokenizer.decode(generated, skip_special_tokens=True)
                boundary = cut_turn(raw, extra_delimiters=("\nQuestion:",))
                if boundary.finish_reason is not None:
                    finish_reason = boundary.finish_reason
                    break
                embeddings = torch.cat((embeddings, self._prefix(
                    torch.tensor([[token]], device=self.device, dtype=torch.long))), dim=1)
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            boundary = cut_turn(raw, extra_delimiters=("\nQuestion:",),
                                allow_partial=finish_reason == "token_limit")
            if boundary.finish_reason is not None:
                finish_reason = boundary.finish_reason
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        if before != self.memory.content_digest():
            raise RuntimeError("Context query modified acquired memory")
        return {
            "text": boundary.text.strip(), "raw_generated_text": raw,
            "mode": mode, "tokens": len(generated), "seconds": time.perf_counter() - started,
            "speaker_source_id": request.speaker_source_id,
            "knowledge_source_id": request.knowledge_source_id,
            "memory_applied": applied, "readout_norm": readout_norm,
            "readout_confidence": confidence,
            "memory_digest": before, "bridge_digest": self.bridge.parameter_digest(),
            "finish_reason": finish_reason, "stop_sequence": boundary.stop_sequence,
            "input_tokens": input_tokens,
            "context_messages_dropped": history_message_count - len(chronological),
            "context_source_ids": tuple(source for source, _ in chronological) + (request.speaker_source_id,),
            "query_context": "current_session_context_v1",
            "prompt_assembly_version": ("role_context_payload_qa_v1" if applied else "message-segments-v1"),
        }

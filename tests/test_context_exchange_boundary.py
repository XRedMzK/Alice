"""Context eviction never presents a detached reply as the first retained turn."""

from test_runtime_bridge import runtime


def test_role_context_drops_an_orphan_assistant_after_its_incoming_turn_does_not_fit():
    app = runtime()
    session = app.store.new_session()
    app.store.add_message(session, "user", "u" * 50)
    app.store.add_message(session, "agent", "previous answer")
    current = app.store.validate_message("user", "new")
    history = app.store.messages(session)
    _inputs, context, dropped = app._prepare(session, history + (current,))
    assert context.source_ids == ("user",) and dropped == 2
    assert context.current_source_id == "user"
    assert app.store.messages(session) == history  # Prompt trimming does not erase UI/M state.


def test_current_explicit_agent_message_is_not_removed_as_an_orphan():
    app = runtime()
    session = app.store.new_session()
    app.store.add_message(session, "user", "u" * 80)
    current = app.store.validate_message("agent", "current request")
    _inputs, context, dropped = app._prepare(session, app.store.messages(session) + (current,))
    assert context.source_ids == ("agent",) and dropped == 1
    assert context.current_source_id == "agent"


def test_initial_agent_message_is_unchanged_without_budget_eviction():
    app = runtime()
    session = app.store.new_session()
    app.store.add_message(session, "agent", "Hello")
    current = app.store.validate_message("user", "Hi")
    _inputs, context, dropped = app._prepare(session, app.store.messages(session) + (current,))
    assert context.source_ids == ("agent", "user") and dropped == 0

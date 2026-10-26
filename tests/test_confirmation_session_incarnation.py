"""Retention must not revive a pending fact under a reused textual session ID."""

import pytest

from test_dialogue_acquisition import EN, action, ask, dialogue, digest


@pytest.mark.parametrize("explicit_id", [False, True], ids=["plain-yes", "old-confirmation-id"])
def test_evicted_session_id_reuse_cannot_confirm_the_former_session_fact(dialogue, explicit_id):
    create, root = dialogue
    app = create()
    session = app.store.new_session("reused")
    incarnation = app.store.session_incarnation(session)
    before = digest(app)
    shown = ask(app, session, EN)
    confirmation_id = shown.memory_info["confirmation_id"]
    app.store.close_session(session)
    for number in range(app.store.MAX_SESSIONS - 1):
        filler = app.store.new_session(f"filler-{number}")
        app.store.close_session(filler)
    app.store.new_session("evicts-former-session")
    assert app.store.retention_status()["drops"]["sessions_evicted"] == 1
    app.store.new_session("reused")
    assert app.store.session_incarnation("reused") != incarnation
    assert len(app.store._incarnations) == app.store.retention_status()["sessions"]
    options = {"confirmation_id": confirmation_id} if explicit_id else {}
    result = ask(app, "reused", "yes", **options)
    assert action(result) == "no_pending"
    assert "window" not in result.text and "confirmation_id" not in result.memory_info
    assert digest(app) == before
    assert not (root / "memory" / "memory.json").exists()
    # The recreated session can still propose and confirm its own new fact.
    current = ask(app, "reused", EN.replace("window", "helmet"))
    assert action(current) == "confirmation_required"
    assert current.memory_info["confirmation_id"] != confirmation_id
    assert action(ask(app, "reused", "yes", confirmation_id=current.memory_info["confirmation_id"])) == "remembered"


def test_session_incarnation_is_stable_in_ram_and_not_added_to_v1_snapshot():
    import json
    from alice_runtime.sessions import SessionStore
    from test_runtime_session_limits import CaptureExport

    store = SessionStore()
    session = store.new_session("stable")
    incarnation = store.session_incarnation(session)
    store.add_message(session, "user", "Current transcript only")
    assert store.session_incarnation(session) == incarnation
    store.clear_session(session)
    assert store.session_incarnation(session) == incarnation
    sink = CaptureExport()
    store.save("snapshot.json", exporter=sink)
    payload = json.loads(sink.content)
    assert set(payload) == {"version", "participants", "sessions"}
    assert set(payload["sessions"][0]) == {"session_id", "closed", "messages"}
    assert incarnation not in sink.content.decode()

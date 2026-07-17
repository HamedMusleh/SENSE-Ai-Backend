"""Safety and sentence-boundary tests for Teta's streaming reply path."""

from types import SimpleNamespace

from ai_pipeline import integration_api
from ai_pipeline.llm import teta_engine
from ai_pipeline.llm.response_strategy import Strategy


def test_high_risk_stream_is_one_hard_coded_chunk(monkeypatch):
    hard_coded_response = "أنا معك. لازم ننادي شخص كبير بنثق فيه هسا."
    strategy = Strategy(
        use_llm=False,
        label="High Risk / Urgent",
        risk_signal="active_death_wish",
        source="hard_coded",
        hard_coded_response=hard_coded_response,
    )
    monkeypatch.setattr(teta_engine, "build_strategy", lambda _: strategy)

    def fail_if_called(**_kwargs):
        raise AssertionError("High Risk must never call the LLM")

    monkeypatch.setattr(teta_engine.client.responses, "create", fail_if_called)

    events = list(
        teta_engine.ask_teta_reply_stream(
            "مش قادر أكمل",
            [],
            {"predicted_label": "High Risk / Urgent"},
        )
    )

    assert [event["type"] for event in events] == ["meta", "sentence", "done"]
    assert events[1]["text"] == hard_coded_response
    assert events[2]["full_text"] == hard_coded_response


def test_llm_stream_splits_sentences_and_caps_history(monkeypatch):
    strategy = Strategy(
        use_llm=True,
        label="Safe / Regulated",
        risk_signal="none",
        source="llm_with_strategy",
        strategy_prompt="Be warm.",
    )
    captured = {}
    fake_stream = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_text.delta", delta="أهلين يا حبيبي."),
        SimpleNamespace(type="response.output_text.delta", delta=" كيفك؟\n"),
        SimpleNamespace(type="response.output_text.delta", delta="أنا هون معك"),
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return fake_stream

    monkeypatch.setattr(teta_engine, "load_prompt", lambda _path: "System prompt")
    monkeypatch.setattr(teta_engine.client.responses, "create", fake_create)

    history = [{"turn": turn, "child_text": f"message-{turn}"} for turn in range(1, 8)]
    events = list(
        teta_engine._call_teta_llm_stream(
            "آخر رسالة",
            history,
            strategy,
        )
    )

    sentences = [event["text"] for event in events if event["type"] == "sentence"]
    assert sentences == ["أهلين يا حبيبي.", "كيفك؟", "أنا هون معك"]
    assert events[-1] == {
        "type": "done",
        "full_text": "أهلين يا حبيبي. كيفك؟\nأنا هون معك",
    }
    assert captured["model"] == "gpt-5"
    assert captured["stream"] is True
    assert "reasoning" not in captured
    assert captured["tools"] == [
        {
            "type": "file_search",
            "vector_store_ids": [teta_engine.SENSE_VECTOR_STORE_ID],
            "max_num_results": 3,
        }
    ]
    user_message = captured["input"][1]["content"]
    assert "message-1" not in user_message
    assert "message-2" in user_message
    assert "message-7" in user_message


def test_non_streaming_llm_also_caps_history(monkeypatch):
    strategy = Strategy(
        use_llm=True,
        label="Safe / Regulated",
        risk_signal="none",
        source="llm_with_strategy",
        strategy_prompt="Be warm.",
    )
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="  رد دافئ  ")

    monkeypatch.setattr(teta_engine, "load_prompt", lambda _path: "System prompt")
    monkeypatch.setattr(teta_engine.client.responses, "create", fake_create)

    history = [{"turn": turn, "child_text": f"message-{turn}"} for turn in range(1, 8)]
    reply = teta_engine._call_teta_llm("آخر رسالة", history, strategy)

    assert reply == "رد دافئ"
    user_message = captured["input"][1]["content"]
    assert "message-1" not in user_message
    assert "message-2" in user_message
    assert "message-7" in user_message


def test_process_turn_stream_high_risk_has_exactly_one_audio_chunk(monkeypatch):
    monkeypatch.setattr(
        integration_api,
        "transcribe_stt",
        lambda _path: ("النص الخام", "مش قادر أكمل"),
    )
    monkeypatch.setattr(
        integration_api,
        "classify_triage",
        lambda *_args, **_kwargs: {
            "predicted_label": "High Risk / Urgent",
            "risk_signal": "active_death_wish",
            "confidence": 1.0,
            "needs_review": True,
        },
    )
    monkeypatch.setattr(
        integration_api,
        "synthesize_openai_tts_to_base64",
        lambda text: f"audio:{text}",
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("High Risk must never call the LLM")

    monkeypatch.setattr(teta_engine.client.responses, "create", fail_if_called)

    events = list(integration_api.process_turn_stream("recording.webm", []))
    chunks = [event for event in events if event["type"] == "audio_chunk"]
    complete = events[-1]

    assert [event["type"] for event in events] == [
        "transcript",
        "audio_chunk",
        "turn_complete",
    ]
    assert len(chunks) == 1
    assert chunks[0]["text"] == complete["reply_text"]
    assert chunks[0]["audio_base64"] == f"audio:{complete['reply_text']}"

from rag_pipeline.generate_response import _conversation_text


def test_conversation_text_keeps_recent_user_and_assistant_turns():
    history = [
        {"role": "system", "content": "hidden"},
        {"role": "user", "content": "Plan five days in Lisbon"},
        {"role": "assistant", "content": "Here is a first itinerary"},
        {"role": "user", "content": "Make it suitable for children"},
    ]

    result = _conversation_text(history)

    assert "System" not in result
    assert "User: Plan five days in Lisbon" in result
    assert "Assistant: Here is a first itinerary" in result
    assert result.endswith("User: Make it suitable for children")


def test_conversation_text_limits_old_messages():
    history = [
        {"role": "user", "content": f"message {number}"}
        for number in range(10)
    ]

    result = _conversation_text(history, max_messages=3)

    assert "message 6" not in result
    assert "message 7" in result
    assert "message 9" in result

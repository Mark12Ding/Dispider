from dispider.conversation import conv_templates


def test_qwen_prompt_matches_released_format():
    conversation = conv_templates["qwen"].copy()
    conversation.append_message("USER", "<image>\nWhat happens?")
    conversation.append_message("ASSISTANT", None)

    assert conversation.get_prompt() == (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions. USER: <image>\nWhat happens? ASSISTANT:"
    )

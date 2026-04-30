from core.training.trainer import sanitize_generation_kwargs


def test_sanitize_generation_kwargs_removes_sampling_args_for_greedy() -> None:
    kwargs = {
        "max_new_tokens": 32,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
    }
    sanitized = sanitize_generation_kwargs(kwargs)
    assert sanitized["max_new_tokens"] == 32
    assert sanitized["do_sample"] is False
    assert "temperature" not in sanitized
    assert "top_p" not in sanitized
    assert "top_k" not in sanitized


def test_sanitize_generation_kwargs_keeps_sampling_args_when_sampling() -> None:
    kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    }
    sanitized = sanitize_generation_kwargs(kwargs)
    assert sanitized["do_sample"] is True
    assert sanitized["temperature"] == 0.7
    assert sanitized["top_p"] == 0.9
    assert sanitized["top_k"] == 50

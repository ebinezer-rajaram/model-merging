from __future__ import annotations

from core.training.train_multitask import _build_task_set_slug, _resolve_mtl_paths


def test_build_task_set_slug_sorted_names_is_order_insensitive() -> None:
    first = _build_task_set_slug(
        ["asr", "emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"],
        mode="sorted_names",
    )
    second = _build_task_set_slug(
        ["vocalsound", "speaker_ver", "langid", "kws", "intent", "emotion", "asr"],
        mode="sorted_names",
    )
    assert first == second
    assert first == "asr_emotion_intent_kws_langid_speaker_ver_vocalsound"


def test_build_task_set_slug_base_then_added_is_order_sensitive() -> None:
    first = _build_task_set_slug(
        ["emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound", "asr"],
        mode="base_then_added",
        base_task_names=["emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"],
        added_task_names=["asr"],
    )
    second = _build_task_set_slug(
        ["kws", "emotion", "intent", "langid", "speaker_ver", "vocalsound", "asr"],
        mode="base_then_added",
        base_task_names=["kws", "emotion", "intent", "langid", "speaker_ver", "vocalsound"],
        added_task_names=["asr"],
    )
    assert first != second
    assert first == "base_emotion_intent_kws_langid_speaker_ver_vocalsound__added_asr"
    assert second == "base_kws_emotion_intent_langid_speaker_ver_vocalsound__added_asr"


def test_build_task_set_slug_base_then_added_empty_side_uses_none() -> None:
    value = _build_task_set_slug(
        ["asr"],
        mode="base_then_added",
        base_task_names=[],
        added_task_names=["asr"],
    )
    assert value == "base_none__added_asr"


def test_resolve_mtl_paths_base_then_added_distinguishes_order(tmp_path) -> None:
    task_names = ["emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound", "asr"]

    first = _resolve_mtl_paths(
        adapter_subdir="test_adapter",
        task_names=task_names,
        layout="task_set",
        task_set_slug_mode="base_then_added",
        base_task_names=["emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"],
        added_task_names=["asr"],
        artifacts_root=tmp_path,
    )
    second = _resolve_mtl_paths(
        adapter_subdir="test_adapter",
        task_names=task_names,
        layout="task_set",
        task_set_slug_mode="base_then_added",
        base_task_names=["kws", "emotion", "intent", "langid", "speaker_ver", "vocalsound"],
        added_task_names=["asr"],
        artifacts_root=tmp_path,
    )
    assert first["base"] != second["base"]
    assert first["task_set_slug"] != second["task_set_slug"]
    assert first["base"].parent == second["base"].parent


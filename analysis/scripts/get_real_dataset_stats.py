"""
Get REAL dataset statistics by using the EXACT functions from train_task.py

This loads datasets exactly as they are loaded during training.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import date

import yaml
from datasets import DatasetDict

DATASET_INFO_PATH = PROJECT_ROOT / "analysis" / "dataset_info.json"


def load_config(task_name):
    """Load config from YAML file."""
    config_path = PROJECT_ROOT / "configs" / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_duration_stats(dataset):
    """Compute duration stats for a dataset split."""
    if dataset is None:
        return {
            "samples": 0,
            "total_seconds": 0.0,
            "hours": 0.0,
            "avg_seconds": None,
        }

    sample_count = len(dataset)
    if sample_count == 0:
        return {
            "samples": 0,
            "total_seconds": 0.0,
            "hours": 0.0,
            "avg_seconds": None,
        }

    if "duration" not in dataset.column_names:
        return {
            "samples": sample_count,
            "total_seconds": 0.0,
            "hours": 0.0,
            "avg_seconds": None,
        }

    durations = dataset["duration"]
    total_seconds = float(sum(durations)) if durations else 0.0
    avg_seconds = (total_seconds / sample_count) if sample_count else None
    return {
        "samples": sample_count,
        "total_seconds": total_seconds,
        "hours": total_seconds / 3600.0,
        "avg_seconds": avg_seconds,
    }


def summarize_task_splits(train, val, test):
    """Summarize train/val/test split stats."""
    train_stats = compute_duration_stats(train)
    val_stats = compute_duration_stats(val)
    test_stats = compute_duration_stats(test)

    total_samples = train_stats["samples"] + val_stats["samples"] + test_stats["samples"]
    total_seconds = train_stats["total_seconds"] + val_stats["total_seconds"] + test_stats["total_seconds"]
    total_hours = total_seconds / 3600.0 if total_seconds else 0.0
    avg_seconds = (total_seconds / total_samples) if total_samples else None

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "total_samples": total_samples,
        "total_hours": total_hours,
        "avg_seconds": avg_seconds,
    }


def _maybe_full(value, use_full_cache):
    return None if use_full_cache else value


def load_asr_datasets(use_full_cache=False):
    """Load ASR datasets with optional full-cache override."""
    from tasks.asr import load_librispeech_subset

    config = load_config("asr")
    ds_cfg = config['dataset']

    train, val, test = load_librispeech_subset(
        train_hours=_maybe_full(ds_cfg.get('train_hours'), use_full_cache),
        val_hours=_maybe_full(ds_cfg.get('val_hours'), use_full_cache),
        test_hours=_maybe_full(ds_cfg.get('test_hours'), use_full_cache),
        test_split=ds_cfg.get('test_split', 'test'),
        return_test_split=True,
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
    )

    return train, val, test, None


def load_emotion_datasets(use_full_cache=False):
    """Load Emotion datasets with optional full-cache override."""
    from tasks.emotion import load_superb_emotion_dataset

    config = load_config("emotion")
    ds_cfg = config['dataset']

    train, val, test, labels = load_superb_emotion_dataset(
        dataset_name=ds_cfg['dataset_name'],
        data_dir=ds_cfg.get('data_dir'),
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        label_column=ds_cfg.get('label_column'),
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
    )

    return train, val, test, labels


def load_intent_datasets(use_full_cache=False):
    """Load Intent datasets with optional full-cache override."""
    from tasks.intent import load_slurp_intent_dataset

    config = load_config("intent")
    ds_cfg = config['dataset']

    train, val, test, labels = load_slurp_intent_dataset(
        dataset_name=ds_cfg['dataset_name'],
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        label_column=ds_cfg.get('label_column'),
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
    )

    return train, val, test, labels


def load_kws_datasets(use_full_cache=False):
    """Load KWS datasets with optional full-cache override."""
    from tasks.kws import load_speech_commands_kws_dataset

    config = load_config("kws")
    ds_cfg = config['dataset']

    train, val, test, labels = load_speech_commands_kws_dataset(
        dataset_name=ds_cfg['dataset_name'],
        dataset_config=ds_cfg.get('dataset_config', 'v0.02'),
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        revision=ds_cfg.get('revision'),
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
    )

    return train, val, test, labels


def load_langid_datasets(use_full_cache=False):
    """Load LangID datasets with optional full-cache override."""
    from tasks.langid import load_fleurs_langid_dataset

    config = load_config("langid")
    ds_cfg = config['dataset']

    train, val, test, labels = load_fleurs_langid_dataset(
        dataset_name=ds_cfg['dataset_name'],
        languages=ds_cfg['languages'],
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
    )

    return train, val, test, labels


def load_speaker_id_datasets(use_full_cache=False):
    """Load Speaker ID datasets with optional full-cache override."""
    from tasks.speaker_id import load_voxceleb_speaker_dataset

    config = load_config("speaker_id")
    ds_cfg = config['dataset']

    train, val, test, labels = load_voxceleb_speaker_dataset(
        dataset_name=ds_cfg['dataset_name'],
        max_speakers=ds_cfg.get('max_speakers'),
        max_samples_per_speaker=ds_cfg.get('max_samples_per_speaker'),
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        label_column=ds_cfg.get('label_column'),
        split_percentages=ds_cfg.get('split_percentages'),
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
        stratify_by_column=ds_cfg.get('stratify_by_column'),
    )

    return train, val, test, labels


def load_speaker_ver_datasets(use_full_cache=False):
    """Load Speaker Verification datasets (pairs) with optional full-cache override."""
    from tasks.speaker_ver import load_speaker_ver_dataset

    config = load_config("speaker_ver")
    ds_cfg = config['dataset']

    train, val, test, labels = load_speaker_ver_dataset(
        dataset_name=ds_cfg['dataset_name'],
        dataset_config=ds_cfg.get('dataset_config'),
        max_train_samples=_maybe_full(ds_cfg.get('max_train_samples'), use_full_cache),
        max_validation_samples=_maybe_full(ds_cfg.get('max_validation_samples'), use_full_cache),
        max_test_samples=_maybe_full(ds_cfg.get('max_test_samples'), use_full_cache),
        max_speakers=ds_cfg.get('max_speakers'),
        pairs_per_speaker=ds_cfg.get('pairs_per_speaker', 200),
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        max_audio_length=ds_cfg.get('max_audio_length'),
        audio_gap_seconds=ds_cfg.get('audio_gap_seconds', 0.5),
        split_by_speakers=ds_cfg.get('split_by_speakers', True),
        seed=ds_cfg['seed'],
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_splits=ds_cfg.get('cache_splits', True),
        force_rebuild=False,
        label_column=ds_cfg.get('label_column'),
        text_column=ds_cfg.get('text_column'),
        audio_column=ds_cfg.get('audio_column'),
        split_percentages=ds_cfg.get('split_percentages'),
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
        stratify_by_column=ds_cfg.get('stratify_by_column'),
        data_dir=ds_cfg.get('data_dir'),
    )

    return train, val, test, labels


def load_speaker_ver_base_stats():
    """Compute base (pre-pair) speaker verification stats."""
    from core.tasks.dataset import add_duration_to_dataset, filter_by_duration, load_and_prepare_dataset
    from tasks.speaker_id.dataset import _select_speakers
    from tasks.speaker_ver.dataset import DATASET_CACHE_ROOT

    config = load_config("speaker_ver")
    ds_cfg = config['dataset']

    dataset, audio_column_name = load_and_prepare_dataset(
        dataset_name=ds_cfg['dataset_name'],
        dataset_config=ds_cfg.get('dataset_config'),
        label_column=ds_cfg.get('label_column'),
        text_column=ds_cfg.get('text_column'),
        audio_column=ds_cfg.get('audio_column'),
        split_percentages=None,
        train_split=ds_cfg.get('train_split', 'train'),
        validation_split=ds_cfg.get('validation_split', 'validation'),
        test_split=ds_cfg.get('test_split', 'test'),
        stratify_by_column=ds_cfg.get('stratify_by_column'),
        seed=ds_cfg['seed'],
        data_dir=ds_cfg.get('data_dir'),
    )

    dataset, selected_speakers = _select_speakers(
        dataset,
        max_speakers=ds_cfg.get('max_speakers'),
        seed=ds_cfg['seed'],
        train_split="train" if "train" in dataset else None,
    )

    base_dataset = dataset.get("train") or next(iter(dataset.values()))
    base_dataset_dict = DatasetDict({"train": base_dataset})

    cache_root = Path(ds_cfg.get("cache_dir")) if ds_cfg.get("cache_dir") else DATASET_CACHE_ROOT
    base_dataset_dict = add_duration_to_dataset(
        base_dataset_dict,
        audio_column=audio_column_name,
        num_proc=ds_cfg.get('num_proc', 'auto'),
        cache_dir=cache_root if not ds_cfg.get('force_rebuild', False) else None,
    )
    base_dataset_dict = filter_by_duration(
        base_dataset_dict,
        max_duration=ds_cfg.get('max_duration'),
        min_duration=ds_cfg.get('min_duration'),
        cache_dir=cache_root if not ds_cfg.get('force_rebuild', False) else None,
    )

    base_dataset = base_dataset_dict["train"]
    num_speakers = len(set(base_dataset["label"])) if "label" in base_dataset.column_names else 0
    return compute_duration_stats(base_dataset), num_speakers, selected_speakers


def print_task_stats(task_name, used_stats, cached_stats=None):
    """Print stats for a single task."""
    print(f"\n{'=' * 80}")
    print(f"{task_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Train:       {used_stats['train']['samples']:>7,} samples  ({used_stats['train']['hours']:>6.2f} hours)")
    print(f"Validation:  {used_stats['val']['samples']:>7,} samples  ({used_stats['val']['hours']:>6.2f} hours)")
    print(f"Test:        {used_stats['test']['samples']:>7,} samples  ({used_stats['test']['hours']:>6.2f} hours)")
    print(
        f"Total:       {used_stats['total_samples']:>7,} samples  ({used_stats['total_hours']:>6.2f} hours)"
    )
    if used_stats.get("classes") is not None:
        print(f"Classes:     {used_stats['classes']}")
    if used_stats.get("label_names") and isinstance(used_stats.get("label_names"), list):
        label_names = used_stats["label_names"]
        if len(label_names) <= 10:
            print(f"Labels:      {label_names}")
        else:
            print(f"Labels:      {label_names[:5]} ... ({len(label_names)} total)")

    if cached_stats:
        print(
            f"Cached:      {cached_stats['total_samples']:>7,} samples  ({cached_stats['total_hours']:>6.2f} hours)"
        )
        if cached_stats["total_samples"] != used_stats["total_samples"]:
            print(
                f"  -> Used:   {used_stats['total_samples']:>7,} samples  ({used_stats['total_hours']:>6.2f} hours)"
            )


def _merge_task_info(existing_info, task_key, updates):
    base = dict(existing_info.get(task_key, {})) if existing_info else {}
    base.update(updates)
    return base


def main():
    """Load all datasets and report stats."""
    print("\n" + "█" * 80)
    print("ACTUAL DATASET STATISTICS (Loaded with your exact training code)")
    print("█" * 80)

    all_stats = {}
    cached_stats_map = {}
    dataset_info = {}

    existing_info = {}
    if DATASET_INFO_PATH.exists():
        with open(DATASET_INFO_PATH, "r") as handle:
            existing_info = json.load(handle)

    tasks = [
        ('asr', 'ASR', load_asr_datasets),
        ('emotion', 'Emotion', load_emotion_datasets),
        ('intent', 'Intent', load_intent_datasets),
        ('kws', 'KWS', load_kws_datasets),
        ('langid', 'LangID', load_langid_datasets),
        ('speaker_id', 'Speaker ID', load_speaker_id_datasets),
        ('speaker_ver', 'Speaker Ver', load_speaker_ver_datasets),
    ]

    for task_key, task_name, func in tasks:
        try:
            print(f"\n⏳ Loading {task_name}...")
            used_train, used_val, used_test, labels = func(use_full_cache=False)
            cached_train, cached_val, cached_test, _ = func(use_full_cache=True)

            used_summary = summarize_task_splits(used_train, used_val, used_test)
            cached_summary = summarize_task_splits(cached_train, cached_val, cached_test)

            used_summary["classes"] = (
                len(labels) if labels is not None else ("Generation" if task_key == "asr" else None)
            )
            if labels is not None:
                used_summary["label_names"] = labels

            all_stats[task_key] = used_summary
            cached_stats_map[task_key] = cached_summary
            print_task_stats(task_name, used_summary, cached_summary)

            dataset_info[task_key] = _merge_task_info(existing_info, task_key, {
                "train_samples": used_summary["train"]["samples"],
                "val_samples": used_summary["val"]["samples"],
                "test_samples": used_summary["test"]["samples"],
                "total_samples": used_summary["total_samples"],
                "train_hours": round(used_summary["train"]["hours"], 2),
                "val_hours": round(used_summary["val"]["hours"], 2),
                "test_hours": round(used_summary["test"]["hours"], 2),
                "total_hours": round(used_summary["total_hours"], 2),
                "num_classes": len(labels) if labels is not None else None,
                "label_names": labels,
                "avg_utterance_seconds": (
                    round(used_summary["avg_seconds"], 3) if used_summary["avg_seconds"] is not None else None
                ),
                "train_avg_utterance_seconds": (
                    round(used_summary["train"]["avg_seconds"], 3)
                    if used_summary["train"]["avg_seconds"] is not None else None
                ),
                "val_avg_utterance_seconds": (
                    round(used_summary["val"]["avg_seconds"], 3)
                    if used_summary["val"]["avg_seconds"] is not None else None
                ),
                "test_avg_utterance_seconds": (
                    round(used_summary["test"]["avg_seconds"], 3)
                    if used_summary["test"]["avg_seconds"] is not None else None
                ),
                "cached_train_samples": cached_summary["train"]["samples"],
                "cached_val_samples": cached_summary["val"]["samples"],
                "cached_test_samples": cached_summary["test"]["samples"],
                "cached_total_samples": cached_summary["total_samples"],
                "cached_train_hours": round(cached_summary["train"]["hours"], 2),
                "cached_val_hours": round(cached_summary["val"]["hours"], 2),
                "cached_test_hours": round(cached_summary["test"]["hours"], 2),
                "cached_total_hours": round(cached_summary["total_hours"], 2),
            })

            if task_key == "asr":
                dataset_info[task_key]["classes"] = "Generation"
        except Exception as e:
            print(f"\n❌ Error loading {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Speaker verification base stats (pre-pairing)
    if "speaker_ver" in dataset_info:
        try:
            base_stats, num_speakers, selected_speakers = load_speaker_ver_base_stats()
            dataset_info["speaker_ver"].update({
                "base_samples": base_stats["samples"],
                "base_hours": round(base_stats["hours"], 2),
                "base_avg_utterance_seconds": (
                    round(base_stats["avg_seconds"], 3) if base_stats["avg_seconds"] is not None else None
                ),
                "base_num_speakers": num_speakers,
                "selected_speakers": selected_speakers,
            })
        except Exception as e:
            print(f"\n❌ Error loading Speaker Ver base stats: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Task':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'Hours':<10} {'Classes':<10}")
    print("-" * 90)

    total_train = 0
    total_val = 0
    total_test = 0
    total_hours = 0.0

    for task_name, stats in all_stats.items():
        train = stats["train"]["samples"]
        val = stats["val"]["samples"]
        test = stats["test"]["samples"]
        hours = stats["total_hours"]
        classes = str(stats.get("classes", "N/A"))

        print(f"{task_name:<15} {train:>9,} {val:>9,} {test:>9,} {train+val+test:>9,} {hours:>9.1f} {classes:>10}")

        total_train += train
        total_val += val
        total_test += test
        total_hours += hours

    print("-" * 90)
    print(f"{'TOTAL':<15} {total_train:>9,} {total_val:>9,} {total_test:>9,} {total_train+total_val+total_test:>9,} {total_hours:>9.1f}")

    if dataset_info:
        summary = {
            "total_samples": total_train + total_val + total_test,
            "total_train": total_train,
            "total_val": total_val,
            "total_test": total_test,
            "total_hours": round(total_hours, 1),
            "num_tasks": len(all_stats),
            "generation_date": date.today().isoformat(),
            "note": "All numbers reflect actual filtered datasets from training code",
            "cached_total_samples": sum(stats["total_samples"] for stats in cached_stats_map.values()),
            "cached_total_hours": round(sum(stats["total_hours"] for stats in cached_stats_map.values()), 1),
        }
        dataset_info = {"summary": summary, **dataset_info}

        DATASET_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_INFO_PATH, "w") as handle:
            json.dump(dataset_info, handle, indent=2)
        print(f"\n✅ Wrote dataset info JSON to: {DATASET_INFO_PATH}")

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

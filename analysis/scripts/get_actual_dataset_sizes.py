"""
Get ACTUAL dataset sizes by loading them with your exact training configs.

This uses your actual dataset loading functions to get the precise numbers
after all filtering, preprocessing, and configuration is applied.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from collections import Counter


def load_config(task_name: str) -> dict:
    """Load the YAML config for a given task."""
    config_path = PROJECT_ROOT / "configs" / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_asr_sizes():
    """Get actual ASR dataset sizes."""
    print("\n" + "=" * 80)
    print("ASR: LibriSpeech")
    print("=" * 80)

    from tasks.asr.dataset import load_librispeech_subset
    config = load_config("asr")

    train, val, test = load_librispeech_subset(
        train_hours=config['dataset'].get('train_hours'),
        val_hours=config['dataset'].get('val_hours'),
        test_hours=config['dataset'].get('test_hours'),
        test_split=config['dataset'].get('test_split', 'test'),
        return_test_split=True,
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")

    return len(train), len(val), len(test)


def get_emotion_sizes():
    """Get actual Emotion dataset sizes."""
    print("\n" + "=" * 80)
    print("Emotion: MELD")
    print("=" * 80)

    from tasks.emotion.dataset import load_emotion_dataset
    config = load_config("emotion")

    train, val, test, label_names = load_emotion_dataset(
        dataset_name=config['dataset']['dataset_name'],
        data_dir=config['dataset'].get('data_dir'),
        max_train_samples=config['dataset'].get('max_train_samples'),
        max_validation_samples=config['dataset'].get('max_validation_samples'),
        max_test_samples=config['dataset'].get('max_test_samples'),
        max_duration=config['dataset'].get('max_duration'),
        min_duration=config['dataset'].get('min_duration'),
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
        label_column=config['dataset'].get('label_column'),
        train_split=config['dataset'].get('train_split', 'train'),
        validation_split=config['dataset'].get('validation_split', 'validation'),
        test_split=config['dataset'].get('test_split', 'test'),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")
    print(f"\nLabels ({len(label_names)}): {label_names}")

    return len(train), len(val), len(test), label_names


def get_intent_sizes():
    """Get actual Intent dataset sizes."""
    print("\n" + "=" * 80)
    print("Intent: SLURP")
    print("=" * 80)

    from tasks.intent.dataset import load_intent_dataset
    config = load_config("intent")

    train, val, test, label_names = load_intent_dataset(
        dataset_name=config['dataset']['dataset_name'],
        max_train_samples=config['dataset'].get('max_train_samples'),
        max_validation_samples=config['dataset'].get('max_validation_samples'),
        max_test_samples=config['dataset'].get('max_test_samples'),
        max_duration=config['dataset'].get('max_duration'),
        min_duration=config['dataset'].get('min_duration'),
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
        label_column=config['dataset'].get('label_column'),
        train_split=config['dataset'].get('train_split', 'train'),
        validation_split=config['dataset'].get('validation_split', 'validation'),
        test_split=config['dataset'].get('test_split', 'test'),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")
    print(f"\nLabels ({len(label_names)}): {label_names[:20]}...")

    return len(train), len(val), len(test), label_names


def get_kws_sizes():
    """Get actual KWS dataset sizes."""
    print("\n" + "=" * 80)
    print("KWS: Speech Commands")
    print("=" * 80)

    from tasks.kws.dataset import load_speech_commands_kws_dataset
    config = load_config("kws")

    train, val, test, label_names = load_speech_commands_kws_dataset(
        dataset_name=config['dataset']['dataset_name'],
        dataset_config=config['dataset'].get('dataset_config', 'v0.02'),
        max_train_samples=config['dataset'].get('max_train_samples'),
        max_validation_samples=config['dataset'].get('max_validation_samples'),
        max_test_samples=config['dataset'].get('max_test_samples'),
        max_duration=config['dataset'].get('max_duration'),
        min_duration=config['dataset'].get('min_duration'),
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
        revision=config['dataset'].get('revision'),
        train_split=config['dataset'].get('train_split', 'train'),
        validation_split=config['dataset'].get('validation_split', 'validation'),
        test_split=config['dataset'].get('test_split', 'test'),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")
    print(f"\nLabels ({len(label_names)}): {label_names}")

    return len(train), len(val), len(test), label_names


def get_langid_sizes():
    """Get actual LangID dataset sizes."""
    print("\n" + "=" * 80)
    print("LangID: FLEURS")
    print("=" * 80)

    from tasks.langid.dataset import load_langid_dataset
    config = load_config("langid")

    train, val, test, label_names = load_langid_dataset(
        dataset_name=config['dataset']['dataset_name'],
        languages=config['dataset']['languages'],
        max_train_samples=config['dataset'].get('max_train_samples'),
        max_validation_samples=config['dataset'].get('max_validation_samples'),
        max_test_samples=config['dataset'].get('max_test_samples'),
        max_duration=config['dataset'].get('max_duration'),
        min_duration=config['dataset'].get('min_duration'),
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
        train_split=config['dataset'].get('train_split', 'train'),
        validation_split=config['dataset'].get('validation_split', 'validation'),
        test_split=config['dataset'].get('test_split', 'test'),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")
    print(f"\nLanguages ({len(label_names)}): {label_names}")

    return len(train), len(val), len(test), label_names


def get_speaker_id_sizes():
    """Get actual Speaker ID dataset sizes."""
    print("\n" + "=" * 80)
    print("Speaker ID: VoxCeleb2")
    print("=" * 80)

    from tasks.speaker_id.dataset import load_speaker_id_dataset
    config = load_config("speaker_id")

    train, val, test, label_names = load_speaker_id_dataset(
        dataset_name=config['dataset']['dataset_name'],
        max_speakers=config['dataset'].get('max_speakers'),
        max_samples_per_speaker=config['dataset'].get('max_samples_per_speaker'),
        max_train_samples=config['dataset'].get('max_train_samples'),
        max_validation_samples=config['dataset'].get('max_validation_samples'),
        max_test_samples=config['dataset'].get('max_test_samples'),
        max_duration=config['dataset'].get('max_duration'),
        min_duration=config['dataset'].get('min_duration'),
        seed=config['dataset']['seed'],
        num_proc=config['dataset'].get('num_proc', 'auto'),
        cache_splits=config['dataset'].get('cache_splits', True),
        force_rebuild=config['dataset'].get('force_rebuild', False),
        label_column=config['dataset'].get('label_column'),
        split_percentages=config['dataset'].get('split_percentages'),
        train_split=config['dataset'].get('train_split', 'train'),
        validation_split=config['dataset'].get('validation_split', 'validation'),
        test_split=config['dataset'].get('test_split', 'test'),
        stratify_by_column=config['dataset'].get('stratify_by_column'),
    )

    print(f"Train:      {len(train):,} samples")
    print(f"Validation: {len(val):,} samples")
    print(f"Test:       {len(test):,} samples")
    print(f"Total:      {len(train) + len(val) + len(test):,} samples")
    print(f"\nNumber of speakers: {len(label_names)}")

    return len(train), len(val), len(test), label_names


def main():
    """Get actual sizes for all datasets."""
    print("\n" + "█" * 80)
    print("ACTUAL DATASET SIZES FROM YOUR TRAINING CONFIGS")
    print("█" * 80)

    results = {}

    try:
        results['asr'] = get_asr_sizes()
    except Exception as e:
        print(f"Error loading ASR: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['emotion'] = get_emotion_sizes()
    except Exception as e:
        print(f"Error loading Emotion: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['intent'] = get_intent_sizes()
    except Exception as e:
        print(f"Error loading Intent: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['kws'] = get_kws_sizes()
    except Exception as e:
        print(f"Error loading KWS: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['langid'] = get_langid_sizes()
    except Exception as e:
        print(f"Error loading LangID: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['speaker_id'] = get_speaker_id_sizes()
    except Exception as e:
        print(f"Error loading Speaker ID: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Task':<15} {'Train':<12} {'Val':<12} {'Test':<12} {'Total':<12} {'Classes':<10}")
    print("-" * 80)

    total_train = 0
    total_val = 0
    total_test = 0

    for task, data in results.items():
        if len(data) >= 3:
            train, val, test = data[0], data[1], data[2]
            total = train + val + test

            # Get num classes
            if len(data) > 3:
                labels = data[3]
                num_classes = len(labels) if labels else "N/A"
            else:
                num_classes = "N/A"

            print(f"{task:<15} {train:>11,} {val:>11,} {test:>11,} {total:>11,} {str(num_classes):>10}")

            total_train += train
            total_val += val
            total_test += test

    print("-" * 80)
    print(f"{'TOTAL':<15} {total_train:>11,} {total_val:>11,} {total_test:>11,} {total_train+total_val+total_test:>11,}")
    print()


if __name__ == "__main__":
    main()

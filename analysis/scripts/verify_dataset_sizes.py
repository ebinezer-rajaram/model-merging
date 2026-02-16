"""
Verify actual dataset sizes by loading the datasets.

This script loads each dataset and reports the actual split sizes
to confirm the information in the dataset summary.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datasets import load_dataset


def load_config(task_name: str) -> dict:
    """Load the YAML config for a given task."""
    config_path = PROJECT_ROOT / "configs" / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def verify_asr():
    """Verify LibriSpeech ASR dataset sizes."""
    print("\n" + "=" * 80)
    print("ASR: LibriSpeech")
    print("=" * 80)
    try:
        ds = load_dataset("librispeech_asr", "clean")
        print(f"Train (train.100):  {len(ds['train.100']):,} samples")
        print(f"Validation:         {len(ds['validation']):,} samples")
        print(f"Test (test.clean):  {len(ds['test']):,} samples")
        print(f"Total:              {len(ds['train.100']) + len(ds['validation']) + len(ds['test']):,} samples")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_intent():
    """Verify SLURP intent dataset sizes."""
    print("\n" + "=" * 80)
    print("Intent: SLURP")
    print("=" * 80)
    try:
        ds = load_dataset("marcel-gohsen/slurp")
        print(f"Train:       {len(ds['train']):,} samples")
        print(f"Validation:  {len(ds['devel']):,} samples")
        print(f"Test:        {len(ds['test']):,} samples")
        print(f"Total:       {len(ds['train']) + len(ds['devel']) + len(ds['test']):,} samples")

        # Sample some intents
        intents = set()
        for i in range(min(1000, len(ds['train']))):
            intents.add(ds['train'][i]['intent'])
        print(f"\nNumber of unique intents: {len(intents)}")
        print(f"Sample intents: {sorted(list(intents))[:10]}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_kws():
    """Verify Speech Commands KWS dataset sizes."""
    print("\n" + "=" * 80)
    print("KWS: Google Speech Commands v0.02")
    print("=" * 80)
    try:
        ds = load_dataset(
            "google/speech_commands",
            data_files={
                "train": "v0.02/train/*.parquet",
                "validation": "v0.02/validation/*.parquet",
                "test": "v0.02/test/*.parquet",
            },
            revision="refs/convert/parquet",
        )
        print(f"Train:       {len(ds['train']):,} samples")
        print(f"Validation:  {len(ds['validation']):,} samples")
        print(f"Test:        {len(ds['test']):,} samples")

        # Filter silence (label 35) to match actual usage
        def filter_silence(split):
            return split.filter(lambda x: x['label'] != 35)

        train_filtered = filter_silence(ds['train'])
        val_filtered = filter_silence(ds['validation'])
        test_filtered = filter_silence(ds['test'])

        print(f"\nAfter filtering silence:")
        print(f"Train:       {len(train_filtered):,} samples")
        print(f"Validation:  {len(val_filtered):,} samples")
        print(f"Test:        {len(test_filtered):,} samples")
        print(f"Total:       {len(train_filtered) + len(val_filtered) + len(test_filtered):,} samples")

        # Get unique labels (excluding silence)
        labels = set()
        for i in range(min(1000, len(train_filtered))):
            labels.add(train_filtered[i]['label'])
        print(f"\nNumber of unique keywords: {len(labels)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_langid():
    """Verify FLEURS language ID dataset sizes."""
    print("\n" + "=" * 80)
    print("LangID: Google FLEURS")
    print("=" * 80)
    try:
        config = load_config("langid")
        languages = config['dataset']['languages']

        lang_mapping = {
            'en': 'en_us',
            'de': 'de_de',
            'fr': 'fr_fr',
            'es': 'es_419',
            'zh': 'cmn_hans_cn',
            'ar': 'ar_eg',
            'hi': 'hi_in',
            'ja': 'ja_jp',
            'ru': 'ru_ru',
            'pt': 'pt_br'
        }

        total_train = 0
        total_val = 0
        total_test = 0

        print(f"Loading {len(languages)} languages...")
        for lang in languages[:3]:  # Sample first 3 languages
            fleurs_code = lang_mapping[lang]
            ds = load_dataset("google/fleurs", fleurs_code)
            print(f"\n{lang} ({fleurs_code}):")
            print(f"  Train: {len(ds['train']):,}")
            print(f"  Val:   {len(ds['validation']):,}")
            print(f"  Test:  {len(ds['test']):,}")
            total_train += len(ds['train'])
            total_val += len(ds['validation'])
            total_test += len(ds['test'])

        # Estimate for all languages
        avg_train = total_train / 3
        avg_val = total_val / 3
        avg_test = total_test / 3

        print(f"\nEstimated total across all {len(languages)} languages:")
        print(f"  Train: ~{int(avg_train * len(languages)):,}")
        print(f"  Val:   ~{int(avg_val * len(languages)):,}")
        print(f"  Test:  ~{int(avg_test * len(languages)):,}")
        print(f"  Total: ~{int((avg_train + avg_val + avg_test) * len(languages)):,}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Verify all dataset sizes."""
    print("\n" + "█" * 80)
    print("DATASET SIZE VERIFICATION")
    print("█" * 80)
    print("\nThis script loads actual datasets to verify the sizes in our summary.")
    print("Some datasets may take time to download/load...")

    results = {}

    print("\n⏳ Verifying ASR (LibriSpeech)...")
    results['asr'] = verify_asr()

    print("\n⏳ Verifying Intent (SLURP)...")
    results['intent'] = verify_intent()

    print("\n⏳ Verifying KWS (Speech Commands)...")
    results['kws'] = verify_kws()

    print("\n⏳ Verifying LangID (FLEURS) - sampling 3 languages...")
    results['langid'] = verify_langid()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nVerification results:")
    for task, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {task:15s}: {status}")

    print("\nNote: MELD and VoxCeleb2 are local/filtered datasets.")
    print("Refer to configs for their exact sizes after preprocessing.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

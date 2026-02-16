"""
Extract comprehensive dataset information from training configs.

This script analyzes your training configurations and datasets to provide:
- Dataset names and sources
- Train/val/test split sizes
- Number of classes/labels
- Language information
- Audio domain characteristics
- Task definitions
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datasets import load_dataset
from collections import Counter
import numpy as np


def load_config(task_name: str) -> dict:
    """Load the YAML config for a given task."""
    config_path = PROJECT_ROOT / "configs" / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def analyze_librispeech(config: dict):
    """Analyze LibriSpeech ASR dataset."""
    print("=" * 80)
    print("TASK: Automatic Speech Recognition (ASR)")
    print("=" * 80)

    print("\nDataset: LibriSpeech ASR")
    print("  Source: https://huggingface.co/datasets/librispeech_asr")
    print("  Subset: clean (train.100 + validation)")

    # Load dataset to get actual sizes
    try:
        ds = load_dataset("librispeech_asr", "clean", split="train.100")
        val = load_dataset("librispeech_asr", "clean", split="validation")
        test = load_dataset("librispeech_asr", "clean", split="test")

        print(f"\nSplit sizes:")
        print(f"  Train: {len(ds):,} samples (~100 hours of audio)")
        print(f"  Validation: {len(val):,} samples (~5 hours of audio)")
        print(f"  Test: {len(test):,} samples (~5 hours of audio)")

        # Get duration info
        if 'audio' in ds.features:
            # Sample a few to estimate duration
            sample_indices = np.random.choice(len(ds), min(100, len(ds)), replace=False)
            durations = []
            for idx in sample_indices:
                audio = ds[int(idx)]['audio']
                if audio and 'array' in audio:
                    duration = len(audio['array']) / audio['sampling_rate']
                    durations.append(duration)

            if durations:
                print(f"\nAudio characteristics (sampled from training set):")
                print(f"  Average duration: {np.mean(durations):.2f} seconds")
                print(f"  Min duration: {np.min(durations):.2f} seconds")
                print(f"  Max duration: {np.max(durations):.2f} seconds")
                print(f"  Sampling rate: {ds[0]['audio']['sampling_rate']:,} Hz")

        print(f"\nLanguage: English only")
        print(f"Task type: Transcription (sequence generation)")
        print(f"Domain: Read audiobook speech, studio-quality, clean")
        print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")

    except Exception as e:
        print(f"  Could not load dataset: {e}")


def analyze_emotion(config: dict):
    """Analyze MELD emotion recognition dataset."""
    print("\n" + "=" * 80)
    print("TASK: Emotion Recognition")
    print("=" * 80)

    print("\nDataset: MELD (Multimodal EmotionLines Dataset)")
    print("  Source: Local dataset (datasets/meld)")
    print("  Reference: https://affective-meld.github.io/")

    # Since it's local, provide known info
    print(f"\nSplit sizes (approximate from MELD paper):")
    print(f"  Train: ~9,989 utterances")
    print(f"  Validation: ~1,109 utterances")
    print(f"  Test: ~2,610 utterances")

    print(f"\nEmotion classes (7):")
    emotions = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
    for i, emotion in enumerate(emotions, 1):
        print(f"  {i}. {emotion}")

    print(f"\nLanguage: English only")
    print(f"Task type: Classification (7-way)")
    print(f"Domain: TV show dialogues (Friends), conversational, emotional expression")
    print(f"Audio characteristics:")
    print(f"  Average duration: ~3 seconds")
    print(f"  Max normal sample: ~30 seconds")
    print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")


def analyze_intent(config: dict):
    """Analyze SLURP intent classification dataset."""
    print("\n" + "=" * 80)
    print("TASK: Intent Classification")
    print("=" * 80)

    print("\nDataset: SLURP (Spoken Language Understanding Resource Package)")
    print("  Source: https://huggingface.co/datasets/marcel-gohsen/slurp")

    try:
        ds = load_dataset("marcel-gohsen/slurp", split="train")
        val = load_dataset("marcel-gohsen/slurp", split="devel")
        test = load_dataset("marcel-gohsen/slurp", split="test")

        print(f"\nSplit sizes:")
        print(f"  Train: {len(ds):,} samples")
        print(f"  Validation (devel): {len(val):,} samples")
        print(f"  Test: {len(test):,} samples")

        # Get intent classes
        if 'intent' in ds.features:
            intents = set()
            for i in range(min(1000, len(ds))):
                intents.add(ds[i]['intent'])

            print(f"\nIntent classes ({len(intents)}):")
            for intent in sorted(intents)[:20]:  # Show first 20
                print(f"  - {intent}")
            if len(intents) > 20:
                print(f"  ... and {len(intents) - 20} more")

        print(f"\nLanguage: English only")
        print(f"Task type: Classification ({len(intents)}-way)")
        print(f"Domain: Smart home/assistant commands, synthetic speech")
        print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")

    except Exception as e:
        print(f"  Could not load dataset: {e}")


def analyze_kws(config: dict):
    """Analyze Google Speech Commands keyword spotting dataset."""
    print("\n" + "=" * 80)
    print("TASK: Keyword Spotting (KWS)")
    print("=" * 80)

    print("\nDataset: Google Speech Commands v0.02")
    print("  Source: https://huggingface.co/datasets/google/speech_commands")

    try:
        ds = load_dataset("google/speech_commands", "v0.02", split="train", revision="refs/convert/parquet")
        val = load_dataset("google/speech_commands", "v0.02", split="validation", revision="refs/convert/parquet")
        test = load_dataset("google/speech_commands", "v0.02", split="test", revision="refs/convert/parquet")

        print(f"\nSplit sizes:")
        print(f"  Train: {len(ds):,} samples")
        print(f"  Validation: {len(val):,} samples")
        print(f"  Test: {len(test):,} samples")

        # Get keywords
        if 'label' in ds.features:
            labels = [ds[i]['label'] for i in range(min(5000, len(ds)))]
            label_counts = Counter(labels)

            print(f"\nKeyword classes (35):")
            for label, count in sorted(label_counts.items()):
                print(f"  - {label}")

        print(f"\nLanguage: English only")
        print(f"Task type: Classification (35-way)")
        print(f"Domain: Short command words, crowdsourced, diverse speakers")
        print(f"Audio characteristics:")
        print(f"  Typical duration: ~1 second")
        print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")

    except Exception as e:
        print(f"  Could not load dataset: {e}")


def analyze_langid(config: dict):
    """Analyze FLEURS language identification dataset."""
    print("\n" + "=" * 80)
    print("TASK: Language Identification (LangID)")
    print("=" * 80)

    print("\nDataset: Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)")
    print("  Source: https://huggingface.co/datasets/google/fleurs")

    languages = config['dataset']['languages']
    lang_mapping = {
        'en': 'en_us (English)',
        'de': 'de_de (German)',
        'fr': 'fr_fr (French)',
        'es': 'es_419 (Spanish)',
        'zh': 'cmn_hans_cn (Mandarin Chinese)',
        'ar': 'ar_eg (Arabic)',
        'hi': 'hi_in (Hindi)',
        'ja': 'ja_jp (Japanese)',
        'ru': 'ru_ru (Russian)',
        'pt': 'pt_br (Portuguese)'
    }

    print(f"\nLanguages ({len(languages)}):")
    for lang in languages:
        print(f"  - {lang_mapping.get(lang, lang)}")

    try:
        # Load one language to get split sizes
        sample_ds = load_dataset("google/fleurs", "en_us")

        print(f"\nSplit sizes (per language):")
        print(f"  Train: ~{len(sample_ds['train']):,} samples per language")
        print(f"  Validation: ~{len(sample_ds['validation']):,} samples per language")
        print(f"  Test: ~{len(sample_ds['test']):,} samples per language")

        print(f"\nTotal across all {len(languages)} languages:")
        print(f"  Train: ~{len(sample_ds['train']) * len(languages):,} samples")
        print(f"  Validation: ~{len(sample_ds['validation']) * len(languages):,} samples")
        print(f"  Test: ~{len(sample_ds['test']) * len(languages):,} samples")

    except Exception as e:
        print(f"  Could not load dataset: {e}")
        print(f"\nSplit sizes (estimated):")
        print(f"  Train: ~2,000 samples per language (20,000 total)")
        print(f"  Validation: ~300 samples per language (3,000 total)")
        print(f"  Test: ~300 samples per language (3,000 total)")

    print(f"\nLanguages: Multilingual (10 languages)")
    print(f"Task type: Classification (10-way)")
    print(f"Domain: Read speech, diverse speakers, parallel corpus")
    print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")


def analyze_speaker_id(config: dict):
    """Analyze VoxCeleb2 speaker identification dataset."""
    print("\n" + "=" * 80)
    print("TASK: Speaker Identification (SpeakerID)")
    print("=" * 80)

    print("\nDataset: VoxCeleb2")
    print("  Source: https://huggingface.co/datasets/acul3/voxceleb2")
    print("  Note: Using top 100 speakers by sample count")

    max_speakers = config['dataset'].get('max_speakers', 100)
    max_samples_per_speaker = config['dataset'].get('max_samples_per_speaker', 200)
    split_pcts = config['dataset']['split_percentages']

    estimated_total = max_speakers * max_samples_per_speaker

    print(f"\nConfiguration:")
    print(f"  Number of speakers: {max_speakers}")
    print(f"  Max samples per speaker: {max_samples_per_speaker}")
    print(f"  Total samples (estimated): {estimated_total:,}")

    print(f"\nSplit sizes (based on {split_pcts}):")
    print(f"  Train: {int(estimated_total * split_pcts['train']):,} samples ({int(split_pcts['train']*100)}%)")
    print(f"  Validation: {int(estimated_total * split_pcts['validation']):,} samples ({int(split_pcts['validation']*100)}%)")
    print(f"  Test: {int(estimated_total * split_pcts['test']):,} samples ({int(split_pcts['test']*100)}%)")

    print(f"\nSpeaker classes: {max_speakers} speakers")
    print(f"\nLanguage: Primarily English (celebrity interviews)")
    print(f"Task type: Classification ({max_speakers}-way)")
    print(f"Domain: Celebrity interviews, YouTube videos, 'in the wild' audio")
    print(f"Audio characteristics:")
    print(f"  Trimmed to: {config['dataset'].get('max_audio_length', 'N/A')} seconds during training")
    print(f"Audio filtering: {config['dataset'].get('min_duration', 'None')}s - {config['dataset'].get('max_duration', 'None')}s")


def main():
    """Extract and display dataset information for all tasks."""

    print("\n" + "█" * 80)
    print("COMPREHENSIVE DATASET ANALYSIS FOR SPEECH MERGING PROJECT")
    print("█" * 80)

    # Analyze each task
    tasks_and_analyzers = [
        ("asr", analyze_librispeech),
        ("emotion", analyze_emotion),
        ("intent", analyze_intent),
        ("kws", analyze_kws),
        ("langid", analyze_langid),
        ("speaker_id", analyze_speaker_id),
    ]

    for task_name, analyzer in tasks_and_analyzers:
        try:
            config = load_config(task_name)
            analyzer(config)
        except Exception as e:
            print(f"\nError analyzing {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Task':<20} {'Dataset':<25} {'Train Size':<15} {'Classes':<10} {'Language':<15}")
    print("-" * 85)
    print(f"{'ASR':<20} {'LibriSpeech':<25} {'~28,539':<15} {'N/A':<10} {'English':<15}")
    print(f"{'Emotion':<20} {'MELD':<25} {'~9,989':<15} {'7':<10} {'English':<15}")
    print(f"{'Intent':<20} {'SLURP':<25} {'~45,000':<15} {'~60':<10} {'English':<15}")
    print(f"{'KWS':<20} {'Speech Commands':<25} {'~75,000':<15} {'35':<10} {'English':<15}")
    print(f"{'LangID':<20} {'FLEURS':<25} {'~20,000':<15} {'10':<10} {'Multilingual':<15}")
    print(f"{'SpeakerID':<20} {'VoxCeleb2':<25} {'~16,000':<15} {'100':<10} {'English':<15}")

    print("\n" + "=" * 80)
    print("AUDIO DOMAIN CHARACTERISTICS")
    print("=" * 80)
    print("\nASR (LibriSpeech):     Studio-quality, read audiobooks, clean")
    print("Emotion (MELD):        Conversational TV dialogue, emotional, acted")
    print("Intent (SLURP):        Synthetic commands, smart home domain")
    print("KWS (Speech Commands): Short words, crowdsourced, diverse conditions")
    print("LangID (FLEURS):       Read speech, multiple languages, parallel corpus")
    print("SpeakerID (VoxCeleb2): Celebrity interviews, 'in the wild', noisy backgrounds")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

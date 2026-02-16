"""Overfit ST on a small fixed subset and evaluate BLEU on the same subset."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict

from transformers import EarlyStoppingCallback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (
    build_early_stopping_kwargs,
    build_training_arguments,
    ensure_dir,
    filter_dataset_columns,
    load_config,
    load_qwen_model,
    parse_training_config,
    run_training_with_evaluation,
    select_random_indices,
    set_global_seed,
)
from core.models.models import create_lora_config_from_dict
from core.training.trainer import CustomTrainer
from experiments.train_task import _resolve_dataset_cache_path
from tasks.st import (
    STCollator,
    compute_st_metrics,
    get_artifact_directories as get_st_artifact_directories,
    load_covost2_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit ST on a small subset.")
    parser.add_argument("--config", default="st.yaml", help="Config filename in configs/")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to overfit")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subset sampling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking subset")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval_steps")
    parser.add_argument("--save-steps", type=int, default=None, help="Override save_steps")
    parser.add_argument(
        "--adapter-suffix",
        default=None,
        help="Optional suffix for adapter subdir (default: overfit_{samples})",
    )
    return parser.parse_args()


def _setup_components(
    config: Dict[str, Any],
    artifact_dirs: Dict[str, Path],
    adapter_subdir: str,
    history_csv: str,
    loss_plot: str,
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    artifacts_cfg = config.get("artifacts", {})
    metrics_cfg = config.get("metrics", {})
    model_cfg = config.get("model", {})

    model_path = PACKAGE_ROOT / model_cfg.get("path", "data/models/Qwen2.5-Omni-3B")
    output_dir = ensure_dir(
        artifact_dirs["adapters"] / artifacts_cfg.get("adapter_subdir", adapter_subdir)
    )
    metrics_dir = ensure_dir(artifact_dirs["metrics"])

    history_csv_path = metrics_dir / metrics_cfg.get("history_csv", history_csv)
    loss_plot_path = metrics_dir / metrics_cfg.get("loss_plot", loss_plot)
    final_adapter_dir = output_dir / "final"

    print("🔧 Loading model and processor...")
    lora_config = None
    if "lora" in model_cfg:
        lora_config = create_lora_config_from_dict(model_cfg["lora"])
        print(
            f"  Using LoRA config from YAML: r={model_cfg['lora'].get('r')}, "
            f"alpha={model_cfg['lora'].get('alpha')}"
        )
    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    return {
        "model": model,
        "processor": processor,
        "output_dir": output_dir,
        "metrics_dir": metrics_dir,
        "history_csv_path": history_csv_path,
        "loss_plot_path": loss_plot_path,
        "final_adapter_dir": final_adapter_dir,
    }


def main() -> None:
    args = parse_args()

    config_path = PACKAGE_ROOT / "configs" / args.config
    config = load_config(config_path)
    seed = int(args.seed or config.get("seed", 0))
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    if args.num_epochs is not None:
        training_cfg["num_train_epochs"] = int(args.num_epochs)
    if args.eval_steps is not None:
        training_cfg["eval_steps"] = int(args.eval_steps)
    if args.save_steps is not None:
        training_cfg["save_steps"] = int(args.save_steps)

    language = config.get("language", "en_de")
    artifact_dirs = get_st_artifact_directories(PACKAGE_ROOT, language=language)

    base_subdir = config.get("artifacts", {}).get("adapter_subdir", "qwen2_5_omni_lora_st")
    suffix = args.adapter_suffix or f"overfit_{args.samples}"
    config.setdefault("artifacts", {})["adapter_subdir"] = f"{base_subdir}_{suffix}"

    components = _setup_components(
        config,
        artifact_dirs,
        adapter_subdir=f"{base_subdir}_{suffix}",
        history_csv="st_training_history.csv",
        loss_plot="st_loss_bleu_plot.png",
    )

    model = components["model"]
    processor = components["processor"]
    output_dir = components["output_dir"]
    metrics_dir = components["metrics_dir"]
    history_csv_path = components["history_csv_path"]
    loss_plot_path = components["loss_plot_path"]
    final_adapter_dir = components["final_adapter_dir"]

    dataset_cache_path = _resolve_dataset_cache_path(dataset_cfg, artifact_dirs)

    loader_kwargs = dict(
        dataset_name=dataset_cfg.get("dataset_name", "fixie-ai/covost2"),
        dataset_config=language,
        max_train_samples=args.samples,
        max_validation_samples=0,
        max_test_samples=0,
        max_duration=dataset_cfg.get("max_duration"),
        min_duration=dataset_cfg.get("min_duration"),
        seed=dataset_cfg.get("seed", seed),
        num_proc=dataset_cfg.get("num_proc"),
        cache_dir=dataset_cache_path,
        cache_splits=False,
        force_rebuild=dataset_cfg.get("force_rebuild", False),
        audio_column=dataset_cfg.get("audio_column"),
        source_column=dataset_cfg.get("source_column"),
        translation_column=dataset_cfg.get("translation_column"),
        train_split=dataset_cfg.get("train_split", "train"),
        validation_split=dataset_cfg.get("validation_split", "validation"),
        test_split=dataset_cfg.get("test_split", "test"),
    )
    train_ds, _, _ = load_covost2_dataset(**loader_kwargs)

    if train_ds is None:
        raise RuntimeError("ST dataset did not provide a training split.")

    keep_columns = ["audio", "text", "translation", "duration"]
    train_ds = filter_dataset_columns(train_ds, keep_columns)

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.shuffle and len(train_ds) > 1:
        indices = select_random_indices(len(train_ds), len(train_ds), seed)
        train_ds = train_ds.select(indices)

    val_ds = train_ds
    full_val_ds = train_ds
    print(f"🎯 Overfit mode: training and eval on {len(train_ds)} train samples (seed={seed}).")

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = STCollator(processor=processor, sampling_rate=target_sr, language_pair=language)

    task_defaults = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "metric_for_best_model": "bleu",
        "greater_is_better": True,
        "early_stopping_patience": 5,
        "length_column_name": "duration",
        "generation_kwargs": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 4,
            "length_penalty": 1.0,
        },
    }
    train_config = parse_training_config(
        training_cfg,
        num_train_examples=len(train_ds),
        task_defaults=task_defaults,
    )

    training_args = build_training_arguments(train_config, output_dir=str(output_dir))
    early_stopping_kwargs = build_early_stopping_kwargs(train_config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=partial(compute_st_metrics, processor=processor),
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
    )

    run_training_with_evaluation(
        trainer,
        processor=processor,
        full_eval_dataset=full_val_ds,
        initial_eval=train_config.initial_eval,
        history_csv_path=history_csv_path,
        loss_plot_path=loss_plot_path,
        final_adapter_dir=final_adapter_dir,
        config=config,
        config_path=config_path,
        metrics_dir=metrics_dir,
        test_dataset=None,
        test_split_name="train",
        resume_from_checkpoint=None,
    )


if __name__ == "__main__":
    main()

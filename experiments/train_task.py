"""Generic task training entry point."""

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path

from transformers import EarlyStoppingCallback, TrainingArguments

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (
    CustomTrainer,
    ensure_dir,
    load_config,
    load_qwen_asr_model,
    plot_loss_and_wer,
    save_artifacts,
    save_history_to_csv,
    set_global_seed,
)
from tasks.asr import (
    compute_asr_metrics,
    get_artifact_directories,
    get_config_path,
    load_librispeech_10h,
    OmniASRCollator,
    TASK_NAME as ASR_TASK_NAME,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train task adapters.")
    parser.add_argument("--task", default=ASR_TASK_NAME, help="Task name to run.")
    parser.add_argument("--config", default=None, help="Override config filename.")
    return parser.parse_args()


def run_asr_task(config_path: Path) -> None:
    """Run the ASR fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    artifacts_cfg = config.get("artifacts", {})
    metrics_cfg = config.get("metrics", {})
    model_cfg = config.get("model", {})

    model_path = PACKAGE_ROOT / model_cfg.get("path", "models/Qwen2.5-Omni-3B")
    artifact_dirs = get_artifact_directories(PACKAGE_ROOT)

    output_dir = ensure_dir(
        artifact_dirs["adapters"] / artifacts_cfg.get("adapter_subdir", "qwen2_5_omni_lora_asr_10h")
    )
    metrics_dir = ensure_dir(artifact_dirs["metrics"])

    history_csv_path = metrics_dir / metrics_cfg.get("history_csv", "training_history.csv")
    loss_plot_path = metrics_dir / metrics_cfg.get("loss_plot", "loss_wer_plot.png")
    final_adapter_dir = output_dir / "final"

    print("🔧 Loading model and processor...")
    model, processor = load_qwen_asr_model(model_path)

    train_ds, val_ds = load_librispeech_10h()

    keep_columns = {"audio", "text"}
    train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in keep_columns])
    val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in keep_columns])

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = OmniASRCollator(processor=processor, sampling_rate=target_sr)

    per_device_train_batch_size = training_cfg.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = training_cfg.get("per_device_eval_batch_size", 4)
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    learning_rate = training_cfg.get("learning_rate", 5e-5)
    lr_scheduler_type = training_cfg.get("lr_scheduler_type", "cosine")
    num_train_epochs = training_cfg.get("num_train_epochs", 2)
    logging_steps = training_cfg.get("logging_steps", 50)
    save_strategy = training_cfg.get("save_strategy", "steps")
    save_steps = training_cfg.get("save_steps", 250)
    save_total_limit = training_cfg.get("save_total_limit", 2)
    eval_strategy = training_cfg.get("eval_strategy", "steps")
    eval_steps = training_cfg.get("eval_steps", 250)
    load_best_model_at_end = training_cfg.get("load_best_model_at_end", True)
    metric_for_best_model = training_cfg.get("metric_for_best_model", "wer")
    greater_is_better = training_cfg.get("greater_is_better", False)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    bf16 = training_cfg.get("bf16", True)
    report_to = training_cfg.get("report_to", ["tensorboard", "wandb"])
    gradient_checkpointing = training_cfg.get("gradient_checkpointing", True)
    gc_kwargs = training_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
    remove_unused_columns = training_cfg.get("remove_unused_columns", False)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.05)
    generation_kwargs = training_cfg.get(
        "generation_kwargs",
        {"max_new_tokens": 128, "do_sample": False},
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_train_examples = len(train_ds)

    updates_per_epoch = math.ceil(
        num_train_examples / (per_device_train_batch_size * gradient_accumulation_steps * world_size)
    )
    total_training_steps = max(1, updates_per_epoch * num_train_epochs)
    warmup_steps = max(1, int(total_training_steps * warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        report_to=report_to,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        remove_unused_columns=remove_unused_columns,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=partial(compute_asr_metrics, processor=processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        generation_kwargs=generation_kwargs,
    )

    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()

    history_rows = save_history_to_csv(trainer, history_csv_path)
    if history_rows:
        plot_loss_and_wer(history_csv_path, loss_plot_path)

    print("💾 Saving LoRA adapter and processor...")
    ensure_dir(final_adapter_dir)
    save_artifacts(trainer, processor, final_adapter_dir)
    print(f"✅ Done: {final_adapter_dir}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.task == ASR_TASK_NAME:
        config_path = get_config_path(PACKAGE_ROOT, args.config)
        run_asr_task(config_path)
    else:
        raise NotImplementedError(f"Task '{args.task}' is not supported yet.")


if __name__ == "__main__":
    main()

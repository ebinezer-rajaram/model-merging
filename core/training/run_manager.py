"""Run management utilities for versioning and tracking training runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunManager:
    """Manages versioned training runs with automatic cleanup and tracking."""

    def __init__(
        self,
        adapter_dir: Path,
        metric_for_ranking: str,
        greater_is_better: bool,
    ):
        """Initialize run manager.

        Args:
            adapter_dir: Base directory for adapter (contains runs/ subdirectory)
            metric_for_ranking: Metric name to use for ranking runs (e.g., "eval_wer")
            greater_is_better: Whether higher metric values are better
        """
        self.adapter_dir = Path(adapter_dir)
        self.runs_dir = self.adapter_dir / "runs"
        self.registry_path = self.adapter_dir / "runs_registry.json"
        self.metric_for_ranking = metric_for_ranking
        self.greater_is_better = greater_is_better

        # Ensure directories exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run_directory(self) -> Path:
        """Create a new timestamped run directory.

        Returns:
            Path to the new run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def register_run(
        self,
        run_dir: Path,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        config_path: Optional[Path] = None,
    ) -> None:
        """Register a completed run and update registry.

        Args:
            run_dir: Path to the run directory
            metrics: Final evaluation metrics for the run
            config: Full configuration dictionary
            config_path: Optional path to original config file (will be copied)
        """
        run_id = run_dir.name

        # Save config with the run
        run_config_path = run_dir / "config.yaml"
        if config_path and config_path.exists():
            shutil.copy(config_path, run_config_path)

        # Save metrics with the run
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # Load existing registry or create new
        registry = self._load_registry()

        # Extract hyperparameters summary
        training_cfg = config.get("training", {})
        lora_cfg = config.get("model", {}).get("lora", {})
        hyperparameters_summary = {
            "learning_rate": training_cfg.get("learning_rate"),
            "lora_r": lora_cfg.get("r"),
            "lora_alpha": lora_cfg.get("alpha"),
            "num_train_epochs": training_cfg.get("num_train_epochs"),
            "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size"),
        }

        # Compute config hash
        config_hash = self._compute_config_hash(config)

        # Create run entry
        run_entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
            "is_best": False,
            "is_latest": True,
            "hyperparameters_summary": hyperparameters_summary,
            "status": "completed",
        }

        # Update registry
        # Mark all runs as not latest
        for run in registry["runs"]:
            run["is_latest"] = False

        # Add new run
        registry["runs"].append(run_entry)
        registry["latest_run_id"] = run_id
        registry["metric_for_ranking"] = self.metric_for_ranking
        registry["greater_is_better"] = self.greater_is_better

        # Rerank runs
        self._rerank_runs(registry)

        # Save registry
        self._save_registry(registry)

        # Update symlinks
        self._update_symlinks(registry)

        # Cleanup old runs
        self._cleanup_old_runs(registry)

    def get_best_run_path(self) -> Optional[Path]:
        """Get path to the best run directory.

        Returns:
            Path to best run or None if no runs exist
        """
        best_link = self.adapter_dir / "best"
        if best_link.exists():
            return best_link.resolve()
        return None

    def get_latest_run_path(self) -> Optional[Path]:
        """Get path to the latest run directory.

        Returns:
            Path to latest run or None if no runs exist
        """
        latest_link = self.adapter_dir / "latest"
        if latest_link.exists():
            return latest_link.resolve()
        return None

    def get_run_path(self, run_id: str) -> Optional[Path]:
        """Get path to a specific run directory.

        Args:
            run_id: Run identifier (e.g., "run_20251109_143022")

        Returns:
            Path to run directory or None if not found
        """
        run_dir = self.runs_dir / run_id
        return run_dir if run_dir.exists() else None

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file or create new."""
        if self.registry_path.exists():
            with self.registry_path.open("r") as f:
                return json.load(f)
        return {
            "metric_for_ranking": self.metric_for_ranking,
            "greater_is_better": self.greater_is_better,
            "runs": [],
            "latest_run_id": None,
        }

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save registry to file."""
        with self.registry_path.open("w") as f:
            json.dump(registry, f, indent=2)

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute MD5 hash of config for identification."""
        import hashlib

        # Convert config to JSON-serializable format (handles Path objects)
        serializable_config = self._make_json_serializable(config)
        config_str = json.dumps(serializable_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _get_ranking_metric_value(self, run: Dict[str, Any]) -> Optional[float]:
        """Extract the ranking metric value from a run entry."""
        metrics = run.get("metrics", {})

        # Try with and without "eval_" prefix
        value = metrics.get(self.metric_for_ranking)
        if value is None and not self.metric_for_ranking.startswith("eval_"):
            value = metrics.get(f"eval_{self.metric_for_ranking}")
        if value is None and self.metric_for_ranking.startswith("eval_"):
            value = metrics.get(self.metric_for_ranking[5:])  # Remove "eval_" prefix

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _rerank_runs(self, registry: Dict[str, Any]) -> None:
        """Rerank all runs and mark the best run."""
        runs = registry["runs"]
        if not runs:
            return

        # Filter runs with valid ranking metric
        ranked_runs = []
        for run in runs:
            metric_value = self._get_ranking_metric_value(run)
            if metric_value is not None:
                ranked_runs.append((run, metric_value))

        if not ranked_runs:
            return

        # Sort runs by metric value
        ranked_runs.sort(
            key=lambda x: x[1],
            reverse=self.greater_is_better,
        )

        # Mark all as not best first
        for run in runs:
            run["is_best"] = False

        # Mark the best run
        if ranked_runs:
            best_run = ranked_runs[0][0]
            best_run["is_best"] = True

    def _update_symlinks(self, registry: Dict[str, Any]) -> None:
        """Update symlinks for best and latest runs."""
        # Remove old symlinks
        best_link = self.adapter_dir / "best"
        latest_link = self.adapter_dir / "latest"

        if best_link.is_symlink() or best_link.exists():
            best_link.unlink(missing_ok=True)
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink(missing_ok=True)

        # Find best and latest runs
        best_run = None
        latest_run = None

        for run in registry["runs"]:
            if run.get("is_best"):
                best_run = run
            if run.get("is_latest"):
                latest_run = run

        # Create symlinks
        if best_run:
            best_run_dir = self.runs_dir / best_run["run_id"]
            if best_run_dir.exists():
                best_link.symlink_to(best_run_dir, target_is_directory=True)

        if latest_run:
            latest_run_dir = self.runs_dir / latest_run["run_id"]
            if latest_run_dir.exists():
                latest_link.symlink_to(latest_run_dir, target_is_directory=True)

    def _cleanup_old_runs(self, registry: Dict[str, Any]) -> None:
        """Remove runs that are not best or latest."""
        runs_to_keep = set()

        for run in registry["runs"]:
            if run.get("is_best") or run.get("is_latest"):
                runs_to_keep.add(run["run_id"])

        # Remove runs not in the keep set
        updated_runs = []
        for run in registry["runs"]:
            run_id = run["run_id"]
            if run_id in runs_to_keep:
                updated_runs.append(run)
            else:
                # Delete run directory
                run_dir = self.runs_dir / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    print(f"🗑️  Removed old run: {run_id}")

        registry["runs"] = updated_runs


def migrate_final_directory(adapter_dir: Path, metric_for_ranking: str, greater_is_better: bool) -> bool:
    """Migrate existing final/ directory to new run structure.

    Args:
        adapter_dir: Base directory for adapter
        metric_for_ranking: Metric name for ranking
        greater_is_better: Whether higher is better

    Returns:
        True if migration was performed, False if no migration needed
    """
    final_dir = adapter_dir / "final"
    if not final_dir.exists():
        return False

    print(f"🔄 Migrating existing final/ directory for {adapter_dir.name}...")

    # Create run manager
    manager = RunManager(adapter_dir, metric_for_ranking, greater_is_better)

    # Create migration run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_MIGRATION_{timestamp}"
    run_dir = manager.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Move contents from final/ to run directory
    for item in final_dir.iterdir():
        dest = run_dir / item.name
        shutil.move(str(item), str(dest))

    # Remove empty final/ directory
    final_dir.rmdir()

    # Try to extract metrics from training_args.bin if available
    metrics = {}
    metrics_json = run_dir / "metrics.json"
    if not metrics_json.exists():
        # Create empty metrics file for migration
        with metrics_json.open("w") as f:
            json.dump({"migrated": True}, f, indent=2)

    # Create basic config placeholder
    config_yaml = run_dir / "config.yaml"
    if not config_yaml.exists():
        with config_yaml.open("w") as f:
            f.write("# Migrated from final/ directory\n")
            f.write("# Original config not available\n")

    # Register the migrated run
    registry = manager._load_registry()
    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config_hash": "migrated",
        "metrics": metrics or {"migrated": True},
        "is_best": True,
        "is_latest": True,
        "hyperparameters_summary": {},
        "status": "migrated",
    }
    registry["runs"].append(run_entry)
    registry["latest_run_id"] = run_id
    registry["metric_for_ranking"] = metric_for_ranking
    registry["greater_is_better"] = greater_is_better
    manager._save_registry(registry)
    manager._update_symlinks(registry)

    print(f"✅ Migration complete: {run_id}")
    return True


__all__ = [
    "RunManager",
    "migrate_final_directory",
]

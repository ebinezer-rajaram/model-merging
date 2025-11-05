"""Configuration loading utilities."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from core.config.schemas import TaskConfig
from core.config.registry import get_task_info, TASK_REGISTRY


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries, overriding base with override values."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_config(config_path: Path, base_config_path: Path | None = None) -> Dict[str, Any]:
    """Load task configuration merged with base defaults."""
    config_path = config_path.resolve()
    config_data = _load_yaml(config_path)

    if base_config_path is None:
        base_config_path = config_path.parent / "base.yaml"

    if base_config_path.exists():
        base_data = _load_yaml(base_config_path)
        return _deep_merge(base_data, config_data)

    return config_data


def get_config_path(task_name: str, config_filename: Optional[str] = None) -> Path:
    """
    Get the path to a task's configuration file.

    Args:
        task_name: Name of the task (must be in TASK_REGISTRY)
        config_filename: Optional custom config filename. If None, uses default from registry.

    Returns:
        Path to the configuration file

    Raises:
        ValueError: If task_name is not registered
    """
    task_info = get_task_info(task_name)

    # Get configs directory (assumes it's at project root / configs)
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    if config_filename is None:
        config_filename = task_info.default_config_file

    return config_dir / config_filename


def load_task_config(
    task_name: str,
    config_filename: Optional[str] = None,
    validate: bool = True,
) -> TaskConfig | Dict[str, Any]:
    """
    Load and optionally validate a task configuration.

    Args:
        task_name: Name of the task
        config_filename: Optional custom config filename
        validate: If True, validates config with Pydantic and returns TaskConfig.
                 If False, returns raw dict.

    Returns:
        TaskConfig object if validate=True, otherwise raw dict

    Raises:
        ValueError: If task is not registered or config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    config_path = get_config_path(task_name, config_filename)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML with base config merging
    config_dict = _load_yaml_config(config_path)

    if not validate:
        return config_dict

    # Validate with Pydantic
    try:
        return TaskConfig(**config_dict)
    except Exception as e:
        raise ValueError(
            f"Invalid configuration for task '{task_name}' in {config_path}: {e}"
        ) from e


def get_artifact_directories(task_name: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get artifact directory paths for a task.

    Args:
        task_name: Name of the task
        base_dir: Base directory for artifacts. If None, uses project root / artifacts / task_name

    Returns:
        Dictionary mapping subdirectory name to Path

    Raises:
        ValueError: If task_name is not registered
    """
    task_info = get_task_info(task_name)

    if base_dir is None:
        # Default: project_root / artifacts / task_name
        project_root = Path(__file__).resolve().parents[2]
        base_dir = project_root / "artifacts" / task_name

    return task_info.get_artifact_dirs(base_dir)

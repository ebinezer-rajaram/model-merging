"""Speaker verification task exports."""

from .config import TASK_NAME, get_artifact_directories, get_config_path
from .dataset import SpeakerVerCollator, load_speaker_ver_dataset
from .metrics import compute_speaker_ver_metrics

__all__ = [
    "load_speaker_ver_dataset",
    "SpeakerVerCollator",
    "TASK_NAME",
    "get_config_path",
    "get_artifact_directories",
    "compute_speaker_ver_metrics",
]

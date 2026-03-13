"""On-disk format and IO helpers for continual compressed merge artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from merging.compression.svd import reconstruct_dense_delta_from_svd

CONTINUAL_SCHEMA_VERSION = "1"
CONTINUAL_MANIFEST_FILE = "continual_artifact_manifest.json"


@dataclass(frozen=True)
class ParamFactorRecord:
    """Manifest entry for one compressed parameter."""

    source_key: str
    original_shape: Tuple[int, ...]
    matrix_shape: Tuple[int, int]
    rank: int
    retained_energy: float
    relative_error: float
    frobenius_norm: float
    shard_file: str
    tensor_key_a: str
    tensor_key_b: str
    scale: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_continual_artifact_dir(path: Path) -> bool:
    return (path / CONTINUAL_MANIFEST_FILE).exists()


def load_continual_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = path / CONTINUAL_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"Continual artifact manifest not found: {manifest_path}")
    with manifest_path.open("r") as handle:
        payload = json.load(handle)
    _validate_manifest(payload, artifact_path=path)
    return payload


def _validate_manifest(payload: Mapping[str, Any], *, artifact_path: Path) -> None:
    if str(payload.get("schema_version")) != CONTINUAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported continual artifact schema_version={payload.get('schema_version')}; "
            f"expected {CONTINUAL_SCHEMA_VERSION}."
        )
    if str(payload.get("artifact_type")) != "continual_compressed_delta":
        raise ValueError("Manifest artifact_type must be 'continual_compressed_delta'.")

    target_params = payload.get("target_params")
    if not isinstance(target_params, list) or not target_params:
        raise ValueError("Manifest target_params must be a non-empty list.")

    for entry in target_params:
        if not isinstance(entry, Mapping):
            raise ValueError("Each target_params entry must be a mapping.")
        source_key = entry.get("source_key")
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("target_params.source_key must be a non-empty string.")
        shard_file = entry.get("shard_file")
        if not isinstance(shard_file, str) or not shard_file:
            raise ValueError(f"target_params[{source_key}] missing shard_file.")
        shard_path = artifact_path / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Factor shard missing for key '{source_key}': {shard_path}")

        rank = int(entry.get("rank", -1))
        if rank < 0:
            raise ValueError(f"target_params[{source_key}] rank must be >= 0.")

        original_shape = entry.get("original_shape")
        if not isinstance(original_shape, list):
            raise ValueError(f"target_params[{source_key}] original_shape must be a list.")


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


class ContinualArtifactWriter:
    """Streaming writer for continual compressed artifacts."""

    def __init__(
        self,
        *,
        output_dir: Path,
        dense_merge_semantics: Dict[str, Any],
        stored_representation: Dict[str, Any],
        provenance_tree: Dict[str, Any],
        constituent_tasks_flat: List[str],
        source_metadata: Optional[List[Dict[str, Any]]] = None,
        shard_max_tensors: int = 2048,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.factor_dir = self.output_dir / "factors"
        self.factor_dir.mkdir(parents=True, exist_ok=True)

        self._dense_merge_semantics = dict(dense_merge_semantics)
        self._stored_representation = dict(stored_representation)
        self._provenance_tree = dict(provenance_tree)
        self._constituent_tasks_flat = list(constituent_tasks_flat)
        self._source_metadata = list(source_metadata) if source_metadata is not None else []

        self._shard_max_tensors = int(max(2, shard_max_tensors))
        self._current_tensors: Dict[str, torch.Tensor] = {}
        self._current_shard_index = 0
        self._target_params: List[Dict[str, Any]] = []
        self._shards: List[Dict[str, Any]] = []

    def _flush_shard(self) -> None:
        if not self._current_tensors:
            return
        shard_name = f"shard_{self._current_shard_index:05d}.safetensors"
        shard_path = self.factor_dir / shard_name
        save_file(self._current_tensors, str(shard_path))
        self._shards.append(
            {
                "shard_file": _safe_relpath(shard_path, self.output_dir),
                "num_tensors": int(len(self._current_tensors)),
            }
        )
        self._current_tensors = {}
        self._current_shard_index += 1

    def add_param(
        self,
        *,
        source_key: str,
        a_factor: torch.Tensor,
        b_factor: torch.Tensor,
        scale: float,
        original_shape: Tuple[int, ...],
        matrix_shape: Tuple[int, int],
        rank: int,
        retained_energy: float,
        relative_error: float,
        frobenius_norm: float,
    ) -> None:
        if not source_key:
            raise ValueError("source_key must be non-empty.")
        if rank < 0:
            raise ValueError(f"rank must be >= 0 for key {source_key}.")
        if a_factor.ndim != 2 or b_factor.ndim != 2:
            raise ValueError(f"Factors for key {source_key} must both be 2D tensors.")

        if len(self._current_tensors) + 2 > self._shard_max_tensors:
            self._flush_shard()

        tensor_prefix = f"p{len(self._target_params):07d}"
        key_a = f"{tensor_prefix}.A"
        key_b = f"{tensor_prefix}.B"
        self._current_tensors[key_a] = a_factor.detach().cpu().contiguous()
        self._current_tensors[key_b] = b_factor.detach().cpu().contiguous()

        shard_name = f"factors/shard_{self._current_shard_index:05d}.safetensors"
        self._target_params.append(
            {
                "source_key": source_key,
                "original_shape": [int(x) for x in original_shape],
                "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
                "rank": int(rank),
                "retained_energy": float(retained_energy),
                "relative_error": float(relative_error),
                "frobenius_norm": float(frobenius_norm),
                "scale": float(scale),
                "shard_file": shard_name,
                "tensor_key_a": key_a,
                "tensor_key_b": key_b,
            }
        )

    def finalize(self, *, diagnostics: Optional[Dict[str, Any]] = None) -> Path:
        self._flush_shard()
        if not self._target_params:
            raise ValueError("Cannot finalize continual artifact with no parameters.")

        manifest = {
            "schema_version": CONTINUAL_SCHEMA_VERSION,
            "artifact_type": "continual_compressed_delta",
            "created_at": _utc_now_iso(),
            "torch_version": str(torch.__version__),
            "dense_merge_semantics": self._dense_merge_semantics,
            "stored_representation": self._stored_representation,
            "source_metadata": self._source_metadata,
            "constituent_tasks_flat": self._constituent_tasks_flat,
            "provenance_tree": self._provenance_tree,
            "target_params": self._target_params,
            "shards": self._shards,
        }
        if diagnostics is not None:
            manifest["diagnostics"] = diagnostics

        manifest_path = self.output_dir / CONTINUAL_MANIFEST_FILE
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest_path


class ContinualArtifactReader:
    """Reader for continual compressed artifact manifests and factor shards."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.manifest = load_continual_manifest(artifact_dir)

        self._entry_by_key: Dict[str, Dict[str, Any]] = {}
        for entry in self.manifest.get("target_params", []):
            key = str(entry["source_key"])
            if key in self._entry_by_key:
                raise ValueError(f"Duplicate source_key in manifest: {key}")
            self._entry_by_key[key] = dict(entry)

    def list_param_keys(self) -> List[str]:
        return sorted(self._entry_by_key.keys())

    def get_entry(self, source_key: str) -> Dict[str, Any]:
        if source_key not in self._entry_by_key:
            raise KeyError(f"Unknown source_key '{source_key}' in continual artifact.")
        return dict(self._entry_by_key[source_key])

    def _load_factor_tensors(self, entry: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        shard_path = self.artifact_dir / str(entry["shard_file"])
        key_a = str(entry["tensor_key_a"])
        key_b = str(entry["tensor_key_b"])
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            a = handle.get_tensor(key_a).clone()
            b = handle.get_tensor(key_b).clone()
        return a, b

    def get_factor_tensors(self, source_key: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        entry = self.get_entry(source_key)
        a, b = self._load_factor_tensors(entry)
        return a, b, float(entry.get("scale", 1.0))

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        entry = self.get_entry(source_key)
        a, b = self._load_factor_tensors(entry)
        original_shape = [int(x) for x in entry["original_shape"]]
        return reconstruct_dense_delta_from_svd(
            a_factor=a,
            b_factor=b,
            scale=float(entry.get("scale", 1.0)),
            original_shape=original_shape,
            dtype=dtype,
            device=device,
        )


__all__ = [
    "CONTINUAL_MANIFEST_FILE",
    "CONTINUAL_SCHEMA_VERSION",
    "ContinualArtifactReader",
    "ContinualArtifactWriter",
    "ParamFactorRecord",
    "is_continual_artifact_dir",
    "load_continual_manifest",
]

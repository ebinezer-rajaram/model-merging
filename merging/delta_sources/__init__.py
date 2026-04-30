"""Unified delta-source abstractions for continual merging."""

from merging.delta_sources.base import DeltaSource, ParamDeltaSpec, ProvenanceNode, SourceMetadata
from merging.delta_sources.lora_source import LoRADeltaSource
from merging.delta_sources.compressed_source import CompressedMergedDeltaSource
from merging.delta_sources.weighted_lora_source import WeightedLoRACompositeSource
from merging.delta_sources.resolver import resolve_delta_source, resolve_delta_sources

__all__ = [
    "CompressedMergedDeltaSource",
    "DeltaSource",
    "LoRADeltaSource",
    "ParamDeltaSpec",
    "ProvenanceNode",
    "SourceMetadata",
    "WeightedLoRACompositeSource",
    "resolve_delta_source",
    "resolve_delta_sources",
]

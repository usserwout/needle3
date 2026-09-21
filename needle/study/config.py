from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple


BASE_COMMIT = "94df9999d58a67ff29f032a41f31307c05554bd6"
DEFAULT_BASE_CHECKPOINT = "checkpoints/needle3.safetensors"
STUDY_SEED = 20260921


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    depth: int
    original_layers: Tuple[int, ...]
    engram_layers: Tuple[int, ...]
    global_layers: Tuple[int, ...]
    attention_gate: str = "elementwise"
    hada_factor_mode: str = "shared"
    engram_bank_ids: Tuple[int, ...] = ()
    cla_pairs: Tuple[Tuple[int, int], ...] = ()
    teacher_run: str | None = None
    requires_training: bool = True
    control_run: str | None = None
    purpose: str = ""

    @property
    def is_reference(self) -> bool:
        return not self.requires_training


RUNS: Dict[str, RunConfig] = {
    "R8": RunConfig(
        run_id="R8",
        depth=8,
        original_layers=(0, 4, 6, 9, 11, 14, 16, 19),
        engram_layers=(4, 7),
        global_layers=(1, 3, 5, 7),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        requires_training=False,
        purpose="Published reference (untouched 8-layer slice)",
    ),
    "C8": RunConfig(
        run_id="C8",
        depth=8,
        original_layers=(0, 4, 6, 9, 11, 14, 16, 19),
        engram_layers=(4, 7),
        global_layers=(1, 3, 5, 7),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        teacher_run="R8",
        requires_training=True,
        control_run="C8",
        purpose="Recovery-trained control for Idea 1 at depth 8",
    ),
    "I1-8": RunConfig(
        run_id="I1-8",
        depth=8,
        original_layers=(0, 4, 6, 9, 11, 14, 16, 19),
        engram_layers=(4, 7),
        global_layers=(1, 3, 5, 7),
        attention_gate="headwise",
        hada_factor_mode="block_untied",
        teacher_run="R8",
        requires_training=True,
        control_run="C8",
        purpose="Test Idea 1 (Headwise gating + block-untied Hadamard MLP)",
    ),
    "R12": RunConfig(
        run_id="R12",
        depth=12,
        original_layers=(0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19),
        engram_layers=(4, 6, 11),
        global_layers=(2, 5, 8, 11),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        requires_training=False,
        purpose="Published reference (untouched 12-layer slice)",
    ),
    "C12": RunConfig(
        run_id="C12",
        depth=12,
        original_layers=(0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19),
        engram_layers=(4, 6, 11),
        global_layers=(2, 5, 8, 11),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        teacher_run="R12",
        requires_training=True,
        control_run="C12",
        purpose="Recovery-trained control for Ideas 2 and 3 at depth 12",
    ),
    "I2-12": RunConfig(
        run_id="I2-12",
        depth=12,
        original_layers=(0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19),
        engram_layers=(4, 6, 11),
        global_layers=(2, 5, 8, 11),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        engram_bank_ids=(0, 0, 1),
        teacher_run="R12",
        requires_training=True,
        control_run="C12",
        purpose="Test Idea 2 (Two shared Engram memory banks across 3 sites)",
    ),
    "I3-12": RunConfig(
        run_id="I3-12",
        depth=12,
        original_layers=(0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19),
        engram_layers=(4, 6, 11),
        global_layers=(2, 5, 8, 11),
        attention_gate="elementwise",
        hada_factor_mode="shared",
        cla_pairs=((0, 1), (3, 4), (6, 7), (9, 10)),
        teacher_run="R12",
        requires_training=True,
        control_run="C12",
        purpose="Test Idea 3 (Cross-Layer Attention across 4 local layer pairs)",
    ),
}


@dataclass
class StudyConfig:
    base_checkpoint: str = DEFAULT_BASE_CHECKPOINT
    base_commit: str = BASE_COMMIT
    output_dir: str = "study_runs"
    manifest_path: str = "study_runs/manifest.json"
    teacher_cache_dir: str = "study_runs/teachers"
    seed: int = STUDY_SEED
    seq_len: int = 512
    tokens_per_batch: int = 8192
    warmup_ratio: float = 0.05
    learning_rate_backbone: float = 3e-5
    learning_rate_heads: float = 1e-5
    learning_rate_engram: float = 3e-6
    weight_decay: float = 0.1
    temperature: float = 2.0
    alpha_ce: float = 0.5
    alpha_kl: float = 0.5
    save_interval_steps: int = 250
    target_token_budget_8l: int = 0
    target_token_budget_12l: int = 0
    sample_size: int = 1000
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    evaluation_paths: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StudyConfig:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

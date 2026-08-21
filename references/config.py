from dataclasses import asdict, dataclass, field
from math import prod
from pathlib import Path
from typing import List, Optional

import yaml

VARIABLES = ["T2M", "Z850", "Z500", "Z250", "T850", "T500", "T250",
             "Q850", "Q500", "Q250", "U850", "U500", "U250", "V850", "V500", "V250"]


@dataclass
class DatasetConfig:
    path: str
    stats_path: Optional[str] = None
    variables: List[str] = field(default_factory=lambda: list(VARIABLES))
    sequence_length: int = 2
    time_slice: Optional[dict] = None
    lat_slice: Optional[dict] = None
    lon_slice: Optional[dict] = None


@dataclass
class NetworkConfig:
    dim: int = 128
    num_layers: int = 4
    num_heads: Optional[int] = None
    dim_heads: int = 32
    expansion_factor: int = 2
    drop_path: float = 0.0


@dataclass
class ObjectiveConfig:
    loss: str = "mse"
    weights: Optional[dict] = None
    kwargs: dict = field(default_factory=lambda: {})


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    batch_size: int = 16
    num_workers: int = 0
    max_steps: int = 2000
    schedulers: list = field(default_factory=lambda: [{"type": "linear", "steps": 100, "start_factor": 0.01},
                                                      {"type": "cosine", "steps": 1900}])
    rollout_steps: int = 4
    train_rollout_steps: int = 1
    pre_steps: int = 0
    val_time_slice: Optional[dict] = None
    val_stride: int = 40
    predict_steps: int = 20
    predict_stride: int = 2


@dataclass
class WorldConfig:
    field_sizes: dict
    patch_sizes: dict

    layout: tuple = field(default_factory=lambda: ())
    token_sizes: dict = field(default_factory=lambda: {})

    field_shape: tuple = field(default_factory=lambda: ())
    token_shape: tuple = field(default_factory=lambda: ())
    patch_shape: tuple = field(default_factory=lambda: ())

    num_tokens: int = field(default_factory=lambda: -1)
    num_elements: int = field(default_factory=lambda: -1)
    dim_tokens: int = field(default_factory=lambda: -1)

    field_pattern: str = field(default_factory=lambda: "")
    token_pattern: str = field(default_factory=lambda: "")
    patch_pattern: str = field(default_factory=lambda: "")
    flat_pattern: str = field(default_factory=lambda: "")

    def __post_init__(self):
        self.layout = tuple(self.field_sizes.keys())

        self.token_sizes = {
            ax: (self.field_sizes[ax] // self.patch_sizes[f"{ax * 2}"])
            for ax in self.layout
        }

        self.field_shape = tuple(self.field_sizes.values())
        self.token_shape = tuple(self.token_sizes.values())
        self.patch_shape = tuple(self.patch_sizes.values())

        self.num_tokens = prod(self.token_sizes.values())
        self.num_elements = prod(self.field_sizes.values())
        self.dim_tokens = prod(self.patch_sizes.values())

        self.field_pattern = " ".join(f"({f} {f * 2})" for f in self.layout)
        self.patch_pattern = " ".join(f"{f * 2}" for f in self.layout)
        self.token_pattern = " ".join(f"{f}" for f in self.layout)
        self.flat_pattern = " ".join(f"{f} {f * 2}" for f in self.layout)


@dataclass
class Config:
    dataset: DatasetConfig
    world: WorldConfig
    network: NetworkConfig = field(default_factory=NetworkConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, cfg: dict) -> "Config":
        return cls(
            dataset=DatasetConfig(**cfg["dataset"]),
            world=WorldConfig(**cfg["world"]),
            network=NetworkConfig(**cfg.get("network", {})),
            objective=ObjectiveConfig(**cfg.get("objective", {})),
            trainer=TrainerConfig(**cfg.get("trainer", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

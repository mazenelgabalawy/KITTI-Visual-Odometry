"""Frozen dataclass schema for pipeline configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    nfeatures: int
    scale_factor: float
    nlevels: int


@dataclass(frozen=True)
class GridFilterConfig:
    grid_size: tuple[int, int]
    max_per_grid: int


@dataclass(frozen=True)
class TemporalMatcherConfig:
    norm: str
    cross_check: bool
    ratio_threshold: float | None = None


@dataclass(frozen=True)
class StereoMatcherConfig:
    kind: str  # "bruteforce" or "rectified"
    cross_check: bool
    pixel_threshold: float = 2.0
    max_hamming_distance: int = 30
    ratio_test: float = 0.7
    max_disparity: float = 64.0

    def __post_init__(self) -> None:
        if self.kind not in ("bruteforce", "rectified"):
            raise ValueError(f"stereo_matcher.kind must be one of 'bruteforce', 'rectified', got {self.kind!r}")


@dataclass(frozen=True)
class LandmarkFilterConfig:
    min_depth_multiplier: float
    relative_depth_uncertainty_threshold: float


@dataclass(frozen=True)
class PnPConfig:
    confidence: float
    iterations_count: int
    seed: int | None = None


@dataclass(frozen=True)
class MotionConfig:
    max_translation: float | None = None
    max_pitch_deg: float | None = None
    orthogonalize_alpha: float | None = None


@dataclass(frozen=True)
class RunConfig:
    sequences: tuple[int, ...]
    dataset_path: str
    output_dir: str


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    detector: DetectorConfig
    temporal_matcher: TemporalMatcherConfig
    stereo_matcher: StereoMatcherConfig
    pnp: PnPConfig
    motion: MotionConfig
    grid_filter: GridFilterConfig | None = None
    landmark_filter: LandmarkFilterConfig | None = None

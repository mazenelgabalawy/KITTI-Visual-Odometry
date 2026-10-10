import dataclasses

import pytest

from kitti_slam.config import ConfigError, load_config
from kitti_slam.config.schema import MotionConfig, PipelineConfig


def test_load_base_yaml_roundtrip(repo_root):
    cfg = load_config(repo_root / "configs" / "base.yaml")

    assert isinstance(cfg, PipelineConfig)
    assert cfg.run.sequences == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert isinstance(cfg.run.sequences, tuple)
    assert cfg.run.dataset_path == "./sequences"
    assert cfg.run.output_dir == "results/base"
    assert cfg.detector.nfeatures == 3000
    assert cfg.detector.scale_factor == 1.6
    assert isinstance(cfg.detector.scale_factor, float)
    assert cfg.detector.nlevels == 8
    assert cfg.temporal_matcher.norm == "NORM_HAMMING"
    assert cfg.temporal_matcher.cross_check is True
    assert cfg.temporal_matcher.ratio_threshold is None
    assert cfg.stereo_matcher.kind == "bruteforce"
    assert cfg.pnp.confidence == 0.99
    assert cfg.pnp.iterations_count == 100
    assert cfg.pnp.seed == 42
    assert cfg.motion == MotionConfig()
    assert cfg.grid_filter is None
    assert cfg.landmark_filter is None


def test_load_constrained_yaml_roundtrip(repo_root):
    cfg = load_config(repo_root / "configs" / "constrained.yaml")

    assert cfg.run.output_dir == "results/constrained"
    assert cfg.detector.scale_factor == 1.2
    assert cfg.grid_filter is not None
    assert cfg.grid_filter.grid_size == (50, 30)
    assert isinstance(cfg.grid_filter.grid_size, tuple)
    assert cfg.grid_filter.max_per_grid == 7
    assert cfg.temporal_matcher.ratio_threshold == 0.5
    assert cfg.stereo_matcher.kind == "rectified"
    assert cfg.stereo_matcher.max_hamming_distance == 50
    assert cfg.stereo_matcher.ratio_test == 0.8
    assert cfg.landmark_filter is not None
    assert cfg.landmark_filter.min_depth_multiplier == 3
    assert cfg.landmark_filter.relative_depth_uncertainty_threshold == 0.5
    assert cfg.pnp.iterations_count == 2000
    assert cfg.motion.max_translation == 15.0
    assert cfg.motion.max_pitch_deg == 30
    assert cfg.motion.orthogonalize_alpha == 0.8


def test_unknown_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
run:
  sequences: [0]
  dataset_path: ./sequences
  output_dir: results/base
  totally_unknown_key: 1
detector:
  nfeatures: 3000
  scale_factor: 1.6
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {}
"""
    )
    with pytest.raises(ConfigError, match=r"run\.totally_unknown_key"):
        load_config(bad)


def test_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
run:
  sequences: [0]
  dataset_path: ./sequences
detector:
  nfeatures: 3000
  scale_factor: 1.6
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {}
"""
    )
    with pytest.raises(ConfigError, match=r"run\.output_dir"):
        load_config(bad)


def test_wrong_type_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
run:
  sequences: [0]
  dataset_path: ./sequences
  output_dir: results/base
detector:
  nfeatures: "not-an-int"
  scale_factor: 1.6
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {}
"""
    )
    with pytest.raises(ConfigError, match=r"detector\.nfeatures"):
        load_config(bad)


@pytest.mark.parametrize(
    ("field_line", "key_path"),
    [
        ("nfeatures: true", r"detector\.nfeatures"),
        ("scale_factor: true", r"detector\.scale_factor"),
    ],
)
def test_bool_not_coerced_to_number(tmp_path, field_line, key_path):
    defaults = {"nfeatures": "nfeatures: 3000", "scale_factor": "scale_factor: 1.6"}
    defaults[field_line.split(":")[0]] = field_line
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"""
run:
  sequences: [0]
  dataset_path: ./sequences
  output_dir: results/base
detector:
  {defaults["nfeatures"]}
  {defaults["scale_factor"]}
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {{}}
"""
    )
    with pytest.raises(ConfigError, match=key_path):
        load_config(bad)


def test_int_coerced_to_float(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(
        """
run:
  sequences: [0]
  dataset_path: ./sequences
  output_dir: results/base
detector:
  nfeatures: 3000
  scale_factor: 2
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 1
  iterations_count: 100
motion: {}
"""
    )
    cfg = load_config(good)
    assert cfg.detector.scale_factor == 2.0
    assert isinstance(cfg.detector.scale_factor, float)
    assert cfg.pnp.confidence == 1.0
    assert isinstance(cfg.pnp.confidence, float)


def test_list_coerced_to_tuple(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(
        """
run:
  sequences: [0, 1, 2]
  dataset_path: ./sequences
  output_dir: results/base
detector:
  nfeatures: 3000
  scale_factor: 1.6
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: bruteforce
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {}
"""
    )
    cfg = load_config(good)
    assert cfg.run.sequences == (0, 1, 2)
    assert isinstance(cfg.run.sequences, tuple)


def test_config_is_frozen(repo_root):
    cfg = load_config(repo_root / "configs" / "base.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run.output_dir = "somewhere_else"


def test_invalid_stereo_kind_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
run:
  sequences: [0]
  dataset_path: ./sequences
  output_dir: results/base
detector:
  nfeatures: 3000
  scale_factor: 1.6
  nlevels: 8
temporal_matcher:
  norm: NORM_HAMMING
  cross_check: true
stereo_matcher:
  kind: not_a_real_kind
  cross_check: true
pnp:
  confidence: 0.99
  iterations_count: 100
motion: {}
"""
    )
    with pytest.raises(ConfigError):
        load_config(bad)

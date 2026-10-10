# Regression tier

These tests need KITTI images, which are not committed. The `kitti_root` fixture
(`./sequences/XX/`) skips with a reason when they are absent. A skip means "not
checked". It never means passed.

| Test | Checks |
|---|---|
| `test_diagnostics.py` | `run_vo.py --diagnostics` writes one row per frame and leaves `poses.txt` byte-identical, on sequence 04 for both configurations |
| `test_determinism.py` | Two runs of the constrained pipeline on sequence 04 produce identical bytes (scoped to the recorded OpenCV/NumPy build and thread settings) |

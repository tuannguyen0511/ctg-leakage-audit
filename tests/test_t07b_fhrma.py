"""Synthetic/no-patient-data checks for P21-T07b."""
from pathlib import Path
import importlib
import numpy as np
import pytest
from scipy.io import savemat

from P21_CTG_Acidemia.src import t07b_fhrma_validate as t


def test_fhr_binary_layout(tmp_path):
    # Timestamp, two FHR channels encoded x4, TOCO encoded x2, reserved byte.
    p = tmp_path / "test01.fhr"
    import struct
    data = struct.pack("<I", 1234567)
    data += struct.pack("<HHBB", 480, 0, 24, 0)
    data += struct.pack("<HHBB", 520, 440, 30, 255)
    p.write_bytes(data)
    f1, f2, uc = t.read_fhr(p)
    np.testing.assert_allclose(f1, [120, 130])
    np.testing.assert_allclose(f2, [0, 110])
    np.testing.assert_allclose(uc, [12, 15])


def test_bad_binary_size_rejected(tmp_path):
    f = tmp_path / "bad.fhr"
    f.write_bytes(b"wrong-length")
    with pytest.raises(ValueError, match="BAD_FHR_SIZE"):
        t.read_fhr(f)


def test_matlab_struct_simplified(tmp_path):
    ann = tmp_path / "expert.mat"
    row = {
        "filename": "train01.fhr", "trainingData": 1,
        "baseline": np.full(240, 121, dtype=float),
        "accelerations": np.array([[1.0, 2.0]]),
        "decelerations": np.empty((0, 2)),
        "unreliableSignal": np.empty((0, 2)),
        "notToAnalyse": np.empty((0, 2)),
    }
    savemat(ann, {"data": np.array([row], dtype=object)})
    d = t.read_annotations(ann)
    assert d["train01.fhr"]["trainingData"] == 1
    assert len(d["train01.fhr"]["baseline"]) == 240
    assert d["train01.fhr"]["accelerations"].shape == (1, 2)
    assert d["train01.fhr"]["decelerations"].shape == (0, 2)


def test_filename_casefold_and_windows_path(tmp_path):
    p = tmp_path / "expert.mat"
    row = dict(filename="C:\\folder\\TRAIN01.FHR", trainingData=1,
               baseline=np.full(4, 120), accelerations=np.empty((0, 2)),
               decelerations=np.empty((0, 2)), unreliableSignal=np.empty((0, 2)),
               notToAnalyse=np.empty((0, 2)))
    savemat(p, {"data": row})
    assert "train01.fhr" in t.read_annotations(p)


def test_interval_shape_and_invalid():
    assert t.intervals([], "x").shape == (0, 2)
    assert t.intervals([0.0, 1.0], "x").shape == (1, 2)
    assert t.intervals(np.array([[0, 2, 4], [1, 3, 5]]), "x").shape == (3, 2)
    with pytest.raises(ValueError, match="INVALID_INTERVALS"):
        t.intervals([[3, 2]], "bad")


def test_mask_half_open():
    a = np.ones(4 * 60 * 3, dtype=bool)
    t.mask_intervals(a, np.asarray([[1., 2.]]))
    assert a[:240].all() and not a[240:480].any() and a[480:].all()


def test_event_boundary_exact_half():
    assert t.match_events([(30, 50)], [(30, 70)]) == (1, 0, 0)
    assert t.match_events([(30, 49)], [(30, 70)]) == (0, 1, 1)


def test_event_overlap_is_expert_denominator():
    # Predicted event is short, but covers exactly 50% of expert event.
    assert t.match_events([(0, 20)], [(0, 40)]) == (1, 0, 0)
    assert t.match_events([(0, 100)], [(70, 90)]) == (1, 0, 0)


def test_event_one_to_one_maximum_cardinality():
    truth = [(0, 40), (60, 100)]
    pred = [(0, 100)]
    assert t.match_events(pred, truth) == (1, 0, 1)
    assert t.match_events([(0, 30), (10, 40)], [(0, 40)]) == (1, 1, 0)
    assert t.match_events([], truth) == (0, 0, 2)


def test_event_valid_fraction():
    valid = np.ones(100, dtype=bool)
    valid[0:35] = False
    assert t.evaluable_events([(0, 100)], valid) == []
    valid[0:30] = False
    valid[30:100] = True
    assert t.evaluable_events([(0, 100)], valid) == [(0, 100)]


def test_original_source_hash():
    # Installed on PC-A from T02; fail if it has been changed.
    file = Path(importlib.import_module("P21_CTG_Acidemia.src.ctg_features").__file__)
    assert file.is_file(), "Original T02 source must already be installed"
    assert t.sha(file) == t.EXPECTED_FEATURE_SHA


def test_wrapper_vs_original_features():
    mod = importlib.import_module("P21_CTG_Acidemia.src.ctg_features")
    cfg = {
        "fs_hz": 4, "baseline_rolling_minutes": 10,
        "baseline_reject_bpm": 15, "event_min_seconds": 15,
        "accel_threshold_bpm": 15, "decel_threshold_bpm": 15,
        "stv_epoch_seconds": 3.75, "stv_min_valid_fraction": 0.5,
        "minute_low_variability_bpm": 5, "prolonged_decel_seconds": 120,
        "contraction_smoothing_seconds": 30,
        "contraction_height_above_median": 15,
        "contraction_min_distance_seconds": 60,
        "uc_nan_limit_pct": 50, "welch_nperseg": 1024,
        "welch_noverlap": 512, "welch_lf_hz": [0.04, 0.15],
        "welch_hf_hz": [0.15, 1.0]
    }
    fhr = np.full(4 * 60 * 10, 120.0)
    fhr[400:500] = 140
    fhr[1000:1100] = 100
    uc = np.full_like(fhr, 20)
    features = mod.compute_features(fhr, uc, cfg)
    assert np.isfinite(features["baseline"])
    events = t.external_wrapper_events(mod, fhr, features["baseline"], cfg)
    assert len(events["accel"]) == features["n_accel"]
    assert len(events["decel"]) == features["n_decel"]
    assert events["accel"] == [(400, 500)]
    assert events["decel"] == [(1000, 1100)]


def test_bootstrap_cluster_and_determinism():
    x = [{"baseline_abs_sum": float(i), "baseline_n": 10,
          "accel_tp": i % 3, "accel_fp": 1, "accel_fn": 2,
          "decel_tp": 1, "decel_fp": 1, "decel_fn": 0}
         for i in range(1, 20)]
    a = t.bootstrap(x, nboot=2000, seed=21)
    b = t.bootstrap(x, nboot=2000, seed=21)
    assert a == b
    assert all(len(a[k]["ci95_cluster_record"]) == 2 for k in a)
    assert all(a[k]["valid_replicates"] == 2000 for k in a)


def test_missing_ground_truth_is_not_counted_as_negative(tmp_path):
    class M: pass
    p = tmp_path / "test01.fhr"
    row, data = t.describe_record(p, "test", None, M(), {})
    assert row["label_status"] == "UNAVAILABLE" and data is None
    assert row["accel_fn"] == "" and row["decel_fn"] == ""
    missing = t.bootstrap([])
    assert all(v["estimate"] is None for v in missing.values())


def test_no_events_f1_undefined():
    assert t.f1(0, 0, 0) is None
    assert t.f1(0, 1, 0) == 0
    assert t.f1(3, 2, 1) == 6 / 9


def test_prereg_and_provenance():
    assert t.PREREG == "85161286492355667fefb0d55d1643cedc5b2736d8234cac1914023c05744c58"
    assert t.N_BOOT == 2000 and t.FS == 4
    assert t.EXPECTED_REPO_COMMIT == "407a279ea55458b0512fd709646c5646dc2ef669"


def test_scoring_pipeline_synthetic_10min_record(tmp_path):
    import struct
    mod = importlib.import_module("P21_CTG_Acidemia.src.ctg_features")
    cfg_path = Path(mod.__file__).resolve().parents[1] / "configs/t02.yaml"
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    n = 10 * 60 * 4
    hr = np.full(n, 120, dtype="<u2")
    hr[400:500] = 140
    hr[1000:1100] = 100
    payload = bytearray(struct.pack("<I", 123456))
    for v in hr:
        payload += struct.pack("<HHBB", int(v) * 4, 0, 40, 0)
    path = tmp_path / "train01.fhr"
    path.write_bytes(payload)
    expert = {"baseline": np.full(n, 120.0),
              "accelerations": np.array([[400 / 240, 500 / 240]]),
              "decelerations": np.array([[1000 / 240, 1100 / 240]]),
              "unreliableSignal": np.empty((0, 2)),
              "notToAnalyse": np.empty((0, 2))}
    row, dat = t.describe_record(path, "train", expert, mod, cfg)
    assert row["label_status"] == "SCORED"
    assert row["baseline_labeled_samples"] == n - 240
    assert row["baseline_mae_bpm"] == 0.0
    assert (row["accel_tp"], row["accel_fp"], row["accel_fn"]) == (1, 0, 0)
    assert (row["decel_tp"], row["decel_fp"], row["decel_fn"]) == (1, 0, 0)
    assert t.aggregate([dat])["accel_f1"] == 1.0


def test_scoring_pipeline_unreliable_and_last_minute(tmp_path):
    import struct
    mod = importlib.import_module("P21_CTG_Acidemia.src.ctg_features")
    cfg_path = Path(mod.__file__).resolve().parents[1] / "configs/t02.yaml"
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    n = 10 * 60 * 4
    payload = bytearray(struct.pack("<I", 123456))
    for _ in range(n):
        payload += struct.pack("<HHBB", 120 * 4, 0, 40, 0)
    path = tmp_path / "train02.fhr"
    path.write_bytes(payload)
    exp = {"baseline": np.full(n, 120.0),
           "accelerations": np.empty((0, 2)),
           "decelerations": np.empty((0, 2)),
           "unreliableSignal": np.array([[0.0, 1.0]]),
           "notToAnalyse": np.array([[2.0, 3.0]])}
    row, dat = t.describe_record(path, "train", exp, mod, cfg)
    assert row["label_status"] == "SCORED"
    assert row["baseline_labeled_samples"] == n - 3 * 60 * 4
    assert dat["accel_tp"] == 0 and dat["accel_fn"] == 0
    assert row["accel_f1"] == ""  # 0/0 undefined, not invented as 1 or 0.


def test_main_partial_outputs_not_falsely_pass(tmp_path, monkeypatch, capsys):
    import struct, csv, json
    mod = importlib.import_module("P21_CTG_Acidemia.src.ctg_features")
    import yaml
    cfg = yaml.safe_load((Path(mod.__file__).resolve().parents[1] / "configs/t02.yaml").read_text())
    n = 10 * 60 * 4
    sig = tmp_path / "train01.fhr"
    payload = bytearray(struct.pack("<I", 112233))
    for _ in range(n):
        payload.extend(struct.pack("<HHBB", 120 * 4, 0, 40, 0))
    sig.write_bytes(payload)
    unavailable_test = tmp_path / "test01.fhr"  # deliberately does not exist: must not read.
    annot = {"baseline": np.full(n, 120.), "accelerations": np.empty((0, 2)),
             "decelerations": np.empty((0, 2)),
             "unreliableSignal": np.empty((0, 2)),
             "notToAnalyse": np.empty((0, 2))}
    details = {"expert_mat_sha256": "f" * 64,
               "external_test_mat_sha256": None,
               "test_labeled_candidates": 0, "train_labeled_candidates": 1}
    monkeypatch.setattr(t, "preflight", lambda *a, **k:
                        (mod, cfg, {sig.name: annot}, [sig], [unavailable_test], details))
    output = tmp_path / "out"
    rc = t.main(["--root", str(tmp_path), "--repo", str(tmp_path),
                 "--output", str(output), "--stage", "run"])
    assert rc == 4
    assert {p.name for p in output.iterdir()} == {"metrics.json", "per_record.csv"}
    result = json.loads((output / "metrics.json").read_text())
    assert result["decision"] == "NOT_ASSESSABLE"
    assert result["splits"]["train"]["scored_count"] == 1
    assert result["splits"]["test"]["scored_count"] == 0
    assert result["splits"]["test"]["metrics"]["accel_f1"]["estimate"] is None
    with (output / "per_record.csv").open(newline="") as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == 2 and records[1]["label_status"] == "UNAVAILABLE"
    # Strict no-overwrite policy.
    with pytest.raises(RuntimeError, match="STOP_OUTPUT_EXISTS"):
        t.main(["--root", str(tmp_path), "--repo", str(tmp_path),
                "--output", str(output), "--stage", "run"])


def test_zero_duration_unreliable_is_true_noop():
    from collections import Counter
    d = Counter()
    got = t.intervals([[0., 0.], [1., 1.], [2., 3.]], "unreliableSignal",
                      record_id="train01.fhr", diagnostics=d)
    assert got.tolist() == [[2., 3.]]
    assert d["unreliableSignal"] == 2
    mask = np.ones(240 * 4, dtype=bool)
    t.mask_intervals(mask, got)
    assert mask[:480].all() and not mask[480:720].any() and mask[720:].all()

def test_zero_duration_not_to_analyse_noop():
    from collections import Counter
    d = Counter()
    z = t.intervals([[2.5, 2.5]], "notToAnalyse", diagnostics=d)
    assert z.shape == (0, 2)
    assert d["notToAnalyse"] == 1

def test_expert_event_zero_duration_must_not_be_removed():
    for field in ("accelerations", "decelerations"):
        with pytest.raises(ValueError, match="zero_duration_event"):
            t.intervals([[1., 1.]], field, record_id="train01.fhr")

def test_mask_nonfinite_negative_reverse_fail_closed():
    for interval, expected in (([[float("nan"), float("nan")]], "nonfinite"),
                               ([[-1., -1.]], "negative_endpoint"),
                               ([[2., 1.]], "reversed_endpoint")):
        with pytest.raises(ValueError, match="INVALID_INTERVALS") as exc:
            t.intervals(interval, "unreliableSignal", record_id="train10.fhr")
        assert expected in str(exc.value)
        assert "train10.fhr" in str(exc.value)

def test_mat_with_zero_duration_mask_succeeds(tmp_path):
    from collections import Counter
    f = tmp_path / "expert.mat"
    row = {"filename": "train02.fhr", "trainingData": 1,
           "baseline": np.full(240, 120.0),
           "accelerations": np.empty((0, 2)),
           "decelerations": np.empty((0, 2)),
           "unreliableSignal": np.array([[0.0, 0.0], [2.0, 3.0]]),
           "notToAnalyse": np.array([[0.0, 0.0]])}
    savemat(f, {"data": np.array([row], dtype=object)})
    diagnostic = Counter()
    a = t.read_annotations(f, diagnostics=diagnostic)
    assert a["train02.fhr"]["unreliableSignal"].tolist() == [[2., 3.]]
    assert a["train02.fhr"]["notToAnalyse"].shape == (0, 2)
    assert sum(diagnostic.values()) == 2


def test_annotation_metadata_audit_ignores_only_zero_duration_masks(tmp_path, capsys):
    from P21_CTG_Acidemia.src import t07b_schema_audit as a
    f = tmp_path / "expert.mat"
    row = {"filename": "train02.fhr", "trainingData": 1,
           "baseline": np.full(240, 120.0),
           "accelerations": np.array([[0.1, 0.2]]),
           "decelerations": np.empty((0, 2)),
           "unreliableSignal": np.array([[0.0, 0.0], [2.0, 3.0]]),
           "notToAnalyse": np.array([[0.0, 0.0]])}
    savemat(f, {"data": np.array([row], dtype=object)})
    assert a.audit_mat(f) == 0
    log = capsys.readouterr().out
    assert "zero_duration_mask=2" in log
    assert "T07B_ANNOTATION_AUDIT=ZERO_DURATION_MASK_ONLY_OR_CLEAN" in log
    row["unreliableSignal"] = np.array([[2.0, 1.0]])
    savemat(f, {"data": np.array([row], dtype=object)})
    assert a.audit_mat(f) == 2
    log = capsys.readouterr().out
    assert "reversed_endpoint=1" in log

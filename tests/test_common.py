import numpy as np
import wfdb
import yaml

from common.io_wfdb import parse_header_comments, read_record
from common.report import render_report, safe_count
from P21_CTG_Acidemia.src import t01_audit


def test_parse_header_comments():
    header = (
        "#-- Outcome measures\n"
        "#pH           7.01\n"
        "#BDecf        14.2\n"
        "#II.stage     -1\n"
        "#Pos. II.st.  -1\n"
        "#Liq. praecox 1\n"
        "#Free text    value with spaces\n"
    )
    result = parse_header_comments(header)
    assert result == {
        "pH": 7.01,
        "BDecf": 14.2,
        "II.stage": -1,
        "Pos. II.st.": -1,
        "Liq. praecox": 1,
        "Free text": "value with spaces",
    }


def test_wfdb_round_trip(tmp_path):
    signal = np.array([[140.0, 30.0], [0.0, 35.0], [150.0, 0.0], [145.0, 45.0]])
    wfdb.wrsamp(
        record_name="synthetic",
        fs=4,
        units=["bpm", "nd"],
        sig_name=["FHR", "UC"],
        p_signal=signal,
        fmt=["16", "16"],
        comments=["pH           7.01", "BDecf        14.2", "II.stage     -1"],
        write_dir=str(tmp_path),
    )
    record = read_record(tmp_path, "synthetic")
    assert record.fs == 4
    assert record.channel_names == ["FHR", "UC"]
    assert record.units == ["bpm", "nd"]
    np.testing.assert_allclose(record.signal, signal, atol=0.01)
    assert record.clinical["pH"] == 7.01
    assert record.clinical["II.stage"] == -1


def test_report_suppression_and_limit():
    assert safe_count(0) == "0"
    assert safe_count(1) == "<10"
    assert safe_count(9) == "<10"
    assert safe_count(10) == "10"
    report = render_report("P21-T01", {"data": "aggregate only"}, {"pytest": True})
    assert "pytest=PASS" in report
    assert len(report.splitlines()) <= 30


def test_audit_outputs_and_no_overwrite(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = [f"synthetic{i:02d}" for i in range(11)]
    (raw / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
    for i, name in enumerate(records):
        wfdb.wrsamp(
            record_name=name,
            fs=4,
            units=["bpm", "nd"],
            sig_name=["FHR", "UC"],
            p_signal=np.array([[140.0, 30.0], [0.0, 35.0], [150.0, 0.0], [145.0, 45.0]]),
            fmt=["16", "16"],
            comments=[f"pH           {7.0 + i / 100:.2f}", "BDecf        14.2", "II.stage     1", "Pos. II.st.  2"],
            write_dir=str(raw),
        )
    config = {
        "run_name": "t01_audit",
        "seed": 42,
        "raw_dir": str(raw),
        "derived_dir": str(tmp_path / "derived"),
        "output_root": str(tmp_path / "outputs"),
        "stage1_windows_minutes": [30, 60],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(t01_audit.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 0})())
    assert t01_audit.audit(config_path) == 0
    report = capsys.readouterr().out
    assert "status: PASS" in report
    assert "records_read_equal_RECORDS=PASS" in report
    assert "n=<10" in report
    assert "pct=<=" in report
    assert len(report.splitlines()) <= 30
    manifest = t01_audit.pd.read_csv(tmp_path / "derived" / "manifest_sha256.csv")
    clinical = t01_audit.pd.read_parquet(tmp_path / "derived" / "clinical.parquet")
    qc = t01_audit.pd.read_parquet(tmp_path / "derived" / "signal_qc.parquet")
    assert len(manifest) == 23
    assert len(clinical) == len(qc) == 11
    assert qc["has_stage2"].all()
    assert (qc["fhr_stage1_last60_n_samples"] == 2).all()
    assert t01_audit.audit(config_path) == 1
    assert "no files overwritten" in capsys.readouterr().out

import json

import pytest

from abeval import extract, pair, read_jsonl
from abeval.__main__ import main


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_read_extract_pair(tmp_path):
    f = tmp_path / "run.jsonl"
    write_jsonl(f, [{"id": "q1", "score": 1}, {"id": "q2", "score": 0}, {"id": "q3", "score": True}])
    values = extract(read_jsonl(f), "score")
    assert values == {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    b = {"q2": 1.0, "q3": 1.0, "q4": 0.0}
    va, vb, dropped = pair(a, b)
    assert va == [0.0, 1.0] and vb == [1.0, 1.0] and dropped == 2


def test_extract_errors():
    with pytest.raises(ValueError):
        extract([{"score": 1}], "score")
    with pytest.raises(ValueError):
        extract([{"id": "a", "score": 1}, {"id": "a", "score": 0}], "score")
    with pytest.raises(ValueError):
        extract([{"id": "a", "score": "high"}], "score")


def test_cli_ci_binary(tmp_path, capsys):
    f = tmp_path / "run.jsonl"
    write_jsonl(f, [{"id": f"q{i}", "score": 1 if i < 8 else 0} for i in range(10)])
    assert main(["ci", str(f)]) == 0
    out = capsys.readouterr().out
    assert "80.0%" in out and "wilson" in out


def test_cli_compare_json(tmp_path, capsys):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, [{"id": f"q{i}", "score": i % 2} for i in range(40)])
    write_jsonl(b, [{"id": f"q{i}", "score": 1 if i % 4 else 0} for i in range(40)])
    assert main(["compare", str(a), str(b), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 40
    assert "p_permutation" in payload and "mcnemar" in payload


def test_cli_compare_no_overlap(tmp_path, capsys):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, [{"id": "x", "score": 1}])
    write_jsonl(b, [{"id": "y", "score": 0}])
    assert main(["compare", str(a), str(b)]) == 2


def test_cli_power(capsys):
    assert main(["power", "--baseline", "0.75", "--delta", "0.03"]) == 0
    out = capsys.readouterr().out
    assert "paired items" in out and "you can detect" in out


def test_cli_reliability(tmp_path, capsys):
    f = tmp_path / "reps.jsonl"
    records = []
    for i in range(6):
        for score in (float(i), float(i), float(i) + (1.0 if i == 0 else 0.0)):
            records.append({"id": f"q{i}", "score": score})
    write_jsonl(f, records)
    assert main(["reliability", str(f)]) == 0
    out = capsys.readouterr().out
    assert "ICC" in out


def test_cli_bad_file(capsys):
    assert main(["ci", "/nonexistent/path.jsonl"]) == 2
    assert "error:" in capsys.readouterr().err

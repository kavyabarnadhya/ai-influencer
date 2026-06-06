"""Tests for scripts/hand_qc.py scoring logic (detectors stubbed — no GPU/model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import hand_qc  # noqa: E402


def _stub(monkeypatch, yolo_n, mp_n, have_mp=True):
    monkeypatch.setattr(hand_qc, "count_hands_yolo", lambda p, conf=0.4: [0.9] * yolo_n)
    monkeypatch.setattr(hand_qc, "count_hands_mp", lambda p: mp_n)
    monkeypatch.setattr(hand_qc, "_HAVE_MP", have_mp)


def test_clean_two_hands(monkeypatch):
    _stub(monkeypatch, yolo_n=2, mp_n=2)
    r = hand_qc.score_image(Path("x.png"))
    assert r["score"] == 0 and not r["flags"]


def test_extra_hands_flagged(monkeypatch):
    _stub(monkeypatch, yolo_n=3, mp_n=3)
    r = hand_qc.score_image(Path("x.png"))
    assert any("EXTRA_HANDS" in f for f in r["flags"])
    assert r["score"] >= 10


def test_deformed_when_mp_fits_fewer(monkeypatch):
    # YOLO sees 1 hand, mediapipe can't model it => likely deformed
    _stub(monkeypatch, yolo_n=1, mp_n=0)
    r = hand_qc.score_image(Path("x.png"))
    assert any("LIKELY_DEFORMED_HAND" in f for f in r["flags"])
    assert r["score"] == 8


def test_no_deformed_signal_without_mediapipe(monkeypatch):
    # MP absent => no finger-level flag, only YOLO count
    _stub(monkeypatch, yolo_n=1, mp_n=0, have_mp=False)
    r = hand_qc.score_image(Path("x.png"))
    assert not any("DEFORMED" in f for f in r["flags"])
    assert r["score"] == 0


def test_expect_hands_zero_flagged(monkeypatch):
    _stub(monkeypatch, yolo_n=0, mp_n=0)
    r = hand_qc.score_image(Path("x.png"), expect_hands=True)
    assert "NO_HANDS_DETECTED" in r["flags"]

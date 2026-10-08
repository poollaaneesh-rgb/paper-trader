import importlib.util
from pathlib import Path

from paper_trader import engines

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("run_daily", ROOT / "scripts" / "run_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_carries_each_accounts_next_weights(tmp_path, monkeypatch):
    rd = _load()
    monkeypatch.setattr(rd, "LIVE", tmp_path)
    name = next(iter(engines.ACCOUNTS))
    states = {name: {"pending": {"SPY": 0.6, "QQQ": 0.0, "TLT": 0.123456}, "bankrupt": None}}
    out = rd.live_summary([], states)
    assert out["accounts"][name]["targets"] == {"SPY": 0.6, "TLT": 0.1235}
    other = next(n for n in engines.ACCOUNTS if n != name)
    assert out["accounts"][other]["targets"] == {}

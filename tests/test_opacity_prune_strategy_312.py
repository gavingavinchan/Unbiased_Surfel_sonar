"""312: opt-in opacity-prune-v1 strategy of native_sonar_training.Session (CPU-only source checks)."""
from pathlib import Path
SRC = (Path(__file__).resolve().parents[1]/'native_sonar_training.py').read_text()


def test_default_strategy_unchanged():
    assert "cfg.get('strategy', 'fixed-density-v1')" in SRC


def test_prune_is_opacity_only_and_after_update():
    body = SRC.split('def _strategy_step')[1].split('def step')[0]
    assert 'opacity_cull' in body and 'densify_and_clone' not in body and 'densify_and_split' not in body
    assert SRC.index('self.model.optimizer.step()') < SRC.index('self._strategy_step()')


def test_strategy_round_trips_in_snapshot():
    assert "'name':self.strategy" in SRC and "state['strategy']['name'] != self.strategy" in SRC

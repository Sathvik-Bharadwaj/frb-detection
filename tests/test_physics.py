import numpy as np
import pytest

from frb.detection import Search
from frb.experiment import threshold_for_fpr
from frb.simulation import Config, dispersion_delay, pulse_template, simulate


def test_delay_units_and_direction():
    # 4148.808 * 100 * (400^-2 - 800^-2) = 1.9448 s, and zero at the reference
    delay = dispersion_delay(100, [400, 800])
    assert delay[0] == pytest.approx(1.94475375)
    assert delay[1] == 0


def test_dedispersion_recovers_injected_arrival_and_dm():
    cfg = Config()
    search = Search(cfg)
    x = pulse_template(cfg, 80, 0.4, 0.012)
    dm_time = x[search.channels, search.indices].sum(axis=1)
    row, t = np.unravel_index(np.argmax(dm_time), dm_time.shape)
    assert search.dms[row] == pytest.approx(80, abs=cfg.dm_step)
    assert t * cfg.dt_s == pytest.approx(0.4, abs=2 * cfg.dt_s)


def test_alignment_does_not_wrap_end_of_record():
    cfg = Config()
    search = Search(cfg)
    assert search.indices.min() >= 0
    assert search.indices.max() < cfg.ntime
    assert np.all(search.indices[:, :, 0] == search.shifts)
    assert search.nout < cfg.ntime


def test_simulation_is_reproducible_and_label_is_not_input():
    cfg = Config()
    x, meta = simulate(cfg, 42, True)
    y, meta2 = simulate(cfg, 42, True)
    assert np.array_equal(x, y) and meta == meta2
    f, _ = Search(cfg).transform(x)
    assert len(f) == 10 and np.isfinite(f).all()


def test_threshold_uses_strict_exceedance():
    scores = np.arange(300)
    threshold = threshold_for_fpr(scores, 0.01)
    assert np.sum(scores > threshold) == 3
    assert np.sum(np.ones(300) > threshold_for_fpr(np.ones(300), 0.01)) == 0


def test_invalid_search_and_input_rejected():
    with pytest.raises(ValueError):
        Search(Config(ntime=16))
    with pytest.raises(ValueError):
        Search(Config()).transform(np.zeros((2, 3)))

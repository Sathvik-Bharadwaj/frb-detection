import h5py
import numpy as np
from frb.observations import read_chime


def test_chime_reader_preserves_mask_and_corrects_frequency_order(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as handle:
        burst = handle.create_group("frb")
        burst["plot_freq"] = [600.0, 700.0]
        burst["plot_time"] = [1.0, 2.0, 3.0]
        burst["wfall"] = [[7.0, np.nan, 9.0], [1.0, 2.0, 3.0]]
        burst["model_wfall"] = [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]
        burst.attrs["tns_name"] = np.bytes_("test")
        burst.attrs["dm"] = 100.0
    observation = read_chime(path)
    assert np.array_equal(observation.intensity[0], [1, 2, 3])
    assert np.isnan(observation.intensity[1, 1])
    assert np.allclose(observation.times_s, [0.001, 0.002, 0.003])


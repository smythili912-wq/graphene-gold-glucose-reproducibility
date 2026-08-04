import numpy as np
import pandas as pd
import pytest

from run_pipeline import synthetic_demo
from src.data_validation import assert_group_disjoint, validate_inputs
from src.simulation import percent_difference, rectangular_channel_hydraulic_resistance
from src.spectral_analysis import add_shift, resonance_features


def test_demo_is_explicitly_synthetic():
    spectra, metadata = synthetic_demo(7)
    assert metadata.is_synthetic.all()
    assert set(metadata.matrix) == {"synthetic_demo"}
    warnings = validate_inputs(spectra, metadata, demo_mode=True)
    assert any("non-invasive" in item for item in warnings)


def test_synthetic_forbidden_in_experimental_mode():
    spectra, metadata = synthetic_demo(7)
    with pytest.raises(ValueError, match="Synthetic"):
        validate_inputs(spectra, metadata, demo_mode=False)


def test_group_leakage_is_blocked():
    with pytest.raises(RuntimeError, match="leakage"):
        assert_group_disjoint(["batch-1", "batch-2"], ["batch-2", "batch-3"])


def test_manuscript_deviation_check():
    assert percent_difference(0.07, 0.08) == pytest.approx(12.5)
    assert abs(0.08 - 0.07) / 0.07 * 100 == pytest.approx(14.285714)


def test_rectangular_channel_resistance_positive():
    value = rectangular_channel_hydraulic_resistance(500e-6, 100e-6, 0.01, 1e-3)
    assert np.isfinite(value) and value > 0


def test_features_are_computable():
    spectra, metadata = synthetic_demo(4)
    features = add_shift(resonance_features(spectra), metadata)
    assert features.shift_nm.notna().all()
    assert features.q_factor.gt(0).all()

from types import SimpleNamespace

import numpy as np

from more_simulation.simulation import Simulation


def test_initial_conditions_preserve_state_order_and_default_missing_values():
    payload = SimpleNamespace(
        stateDescription=[
            SimpleNamespace(size=2, ic=[1.5, -2.0]),
            SimpleNamespace(size=3, ic=[]),
            SimpleNamespace(size=1, ic=[4.0]),
        ]
    )

    initial = Simulation._extract_initial_conditions(payload)

    np.testing.assert_allclose(np.asarray(initial).ravel(), [1.5, -2, 0, 0, 0, 4])


def test_initial_conditions_allow_a_stateless_model():
    payload = SimpleNamespace(stateDescription=[])

    assert Simulation._extract_initial_conditions(payload).numel() == 0

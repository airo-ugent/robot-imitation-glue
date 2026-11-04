"""the UR3 eval preprocessor gives the policy the inputs of the prepared dataset (wrist image + EEF state)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_imitation_glue.ur3station.eval_lerobot import wrist_preprocessor  # noqa: E402


def test_wrist_preprocessor():
    obs = {
        "wrist_image": np.full((224, 224, 3), 255, dtype=np.uint8),
        "state": np.arange(7, dtype=np.float32),
        "joints": np.zeros(6, dtype=np.float32),
    }
    policy_obs = wrist_preprocessor(obs)

    assert set(policy_obs) == {"observation.images.wrist_image", "observation.state"}
    image = policy_obs["observation.images.wrist_image"]
    assert image.shape == (1, 3, 224, 224) and image.dtype == torch.float32
    assert float(image.max()) == pytest.approx(1.0)
    np.testing.assert_allclose(policy_obs["observation.state"].numpy(), [np.arange(7)])
    # a fresh dict: the observation is recorded after get_action
    assert set(obs) == {"wrist_image", "state", "joints"}

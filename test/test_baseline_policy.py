import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from robot_imitation_glue.baseline.agent import BaselineAgent  # noqa: E402
from robot_imitation_glue.baseline.policy import (  # noqa: E402
    PROPRIO_CONSTANT_STD,
    BCPolicy,
    Normalizer,
    load_checkpoint,
    random_shift,
    save_checkpoint,
    to_chw_float_image,
)

CAMERA_KEYS = ["observation.images.scene_image", "observation.images.wrist_image"]
PROPRIO_DIM = 7
ACTION_DIM = 7
IMAGE_SIZE = 64


def make_policy():
    return BCPolicy(camera_keys=CAMERA_KEYS, proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)


def test_policy_forward_pass():
    policy = make_policy()
    batch_size = 2
    images = {key: torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE) for key in CAMERA_KEYS}
    proprio = torch.randn(batch_size, PROPRIO_DIM)

    action = policy(images, proprio)
    assert action.shape == (batch_size, ACTION_DIM)

    loss = policy.compute_loss(images, proprio, torch.randn(batch_size, ACTION_DIM))
    assert loss.ndim == 0 and loss.item() >= 0.0


def test_normalizer_roundtrip():
    normalizer = Normalizer(torch.arange(3).float(), torch.ones(3) * 2)
    x = torch.randn(5, 3)
    assert torch.allclose(normalizer.unnormalize(normalizer.normalize(x)), x, atol=1e-5)


def test_proprio_normalizer_only_centers_constant_dims():
    # dim 0 was constant in the training data, dim 1 varies a little, dim 2 normally
    std = torch.tensor([1e-7, 5e-3, 0.5])
    normalizer = Normalizer(torch.zeros(3), std, constant_std=PROPRIO_CONSTANT_STD)
    assert torch.equal(normalizer.std, torch.tensor([1.0, 5e-3, 0.5]))
    # 1 mm of sensor noise on the constant dim stays a small input instead of 1e4
    assert normalizer.normalize(torch.tensor([1e-3, 0.0, 0.0]))[0].abs() < 1e-2


def test_action_normalizer_keeps_small_stds():
    # without constant_std (actions) a small but real std is kept, so unnormalized errors stay at that scale
    normalizer = Normalizer(torch.zeros(2), torch.tensor([1e-4, 5e-3]))
    assert torch.allclose(normalizer.std, torch.tensor([1e-4, 5e-3]))


def test_random_shift_keeps_size_and_shifts_content():
    images = torch.rand(8, 3, 16, 16)
    assert torch.equal(random_shift(images, 0), images)
    shifted = random_shift(images, 2)
    assert shifted.shape == images.shape
    # every output is the input translated by at most 2 pixels: the inner 12x12 region appears unchanged in it
    for image, out in zip(images, shifted):
        assert any(
            torch.equal(out[:, 2 + dy : 14 + dy, 2 + dx : 14 + dx], image[:, 2:14, 2:14])
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        )


def test_agent_get_action_on_synthetic_observation(tmp_path):
    policy = make_policy()
    checkpoint_path = tmp_path / "policy.ckpt"
    save_checkpoint(
        checkpoint_path,
        policy,
        Normalizer(torch.zeros(PROPRIO_DIM), torch.ones(PROPRIO_DIM)),
        Normalizer(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
        IMAGE_SIZE,
    )
    loaded_policy, _, _, image_size = load_checkpoint(checkpoint_path)
    assert image_size == IMAGE_SIZE
    assert loaded_policy.camera_keys == CAMERA_KEYS

    agent = BaselineAgent(
        checkpoint_path,
        camera_key_mapping={"img1": CAMERA_KEYS[0], "img2": CAMERA_KEYS[1]},
        state_key="state",
    )
    observation = {
        # images are HWC uint8 and of a different resolution than the policy input, as in the real environments
        "img1": np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8),
        "img2": np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8),
        "state": np.random.randn(PROPRIO_DIM).astype(np.float32),
    }
    action = agent.get_action(observation)
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(action))


def test_image_scaling_depends_on_dtype_not_content():
    # a (nearly) black uint8 frame must be scaled like any other uint8 frame
    dark = torch.ones((4, 4, 3), dtype=torch.uint8)
    assert to_chw_float_image(dark).max().item() == pytest.approx(1 / 255)
    # float images are already in [0, 1]
    bright = torch.full((3, 4, 4), 0.5)
    assert to_chw_float_image(bright).max().item() == pytest.approx(0.5)

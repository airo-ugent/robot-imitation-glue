import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from robot_imitation_glue.baseline.dit_agent import DiTAgent  # noqa: E402
from robot_imitation_glue.baseline.dit_policy import DiTPolicy  # noqa: E402
from robot_imitation_glue.baseline.policy import Normalizer, load_checkpoint, save_checkpoint  # noqa: E402

CAMERA_KEYS = ["observation.images.scene_image", "observation.images.wrist_image"]
PROPRIO_DIM = 7
ACTION_DIM = 7
IMAGE_SIZE = 64
HORIZON = 4


def make_policy():
    # tiny model so the tests run on cpu in seconds
    return DiTPolicy(
        camera_keys=CAMERA_KEYS,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        prediction_horizon=HORIZON,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
    )


def make_batch(batch_size=2):
    images = {key: torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE) for key in CAMERA_KEYS}
    proprio = torch.randn(batch_size, PROPRIO_DIM)
    actions = torch.randn(batch_size, HORIZON, ACTION_DIM)
    return images, proprio, actions


def make_observation():
    return {
        # images are HWC uint8 and of a different resolution than the policy input, as in the real environments
        "img1": np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8),
        "img2": np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8),
        "state": np.random.randn(PROPRIO_DIM).astype(np.float32),
    }


def make_agent(tmp_path, **kwargs):
    checkpoint_path = tmp_path / "policy.ckpt"
    save_checkpoint(
        checkpoint_path,
        make_policy(),
        Normalizer(torch.zeros(PROPRIO_DIM), torch.ones(PROPRIO_DIM)),
        Normalizer(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
        IMAGE_SIZE,
    )
    loaded_policy, _, _, image_size, _ = load_checkpoint(checkpoint_path, policy_class=DiTPolicy)
    assert image_size == IMAGE_SIZE
    assert loaded_policy.camera_keys == CAMERA_KEYS
    return DiTAgent(
        checkpoint_path,
        camera_key_mapping={"img1": CAMERA_KEYS[0], "img2": CAMERA_KEYS[1]},
        state_key="state",
        n_sampling_steps=2,
        **kwargs,
    )


def test_policy_forward_pass():
    policy = make_policy()
    images, proprio, actions = make_batch()
    t = torch.rand(actions.shape[0])

    velocity = policy(actions, t, images, proprio)
    assert velocity.shape == actions.shape
    assert torch.all(torch.isfinite(velocity))


def test_adaln_zero_initialization_makes_the_blocks_identity():
    """the zero-initialized modulation projections are what makes it adaLN-*Zero*."""
    policy = make_policy()
    for block in policy.blocks:
        modulation = block.adaln_modulation[1]
        assert torch.all(modulation.weight == 0) and torch.all(modulation.bias == 0)
        hidden_dim = policy.config["hidden_dim"]
        x = torch.randn(2, HORIZON, hidden_dim)
        assert torch.allclose(block(x, torch.randn(2, hidden_dim)), x)
    # the output head is zero-initialized as well, so the predicted velocity starts at zero
    images, proprio, actions = make_batch()
    assert torch.all(policy(actions, torch.rand(2), images, proprio) == 0)


def test_flow_matching_loss_is_a_finite_scalar_with_gradients():
    policy = make_policy()
    images, proprio, actions = make_batch()

    loss = policy.compute_loss(images, proprio, actions)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() >= 0.0

    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients, "the loss did not produce any gradient"
    assert any(torch.any(g != 0) for g in gradients)
    assert all(torch.all(torch.isfinite(g)) for g in gradients)


def test_loss_only_averages_over_the_in_episode_timesteps():
    policy = make_policy()
    images, proprio, actions = make_batch(batch_size=2)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    torch.manual_seed(0)
    masked_loss = policy.compute_loss(images, proprio, actions, mask)

    # replay the same flow time and noise and check the masked mean by hand
    torch.manual_seed(0)
    t = torch.rand(actions.shape[0])
    noise = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
    error = (policy(x_t, t, images, proprio) - (actions - noise)) ** 2
    expected = error[mask.bool()].mean()
    assert torch.allclose(masked_loss, expected, atol=1e-6)

    # an all-ones mask must be identical to no mask at all (same scale, not summed over the horizon)
    torch.manual_seed(0)
    unmasked_loss = policy.compute_loss(images, proprio, actions)
    torch.manual_seed(0)
    assert torch.allclose(unmasked_loss, policy.compute_loss(images, proprio, actions, torch.ones_like(mask)))


def test_sampler_returns_an_action_chunk():
    policy = make_policy()
    images, proprio, _ = make_batch(batch_size=3)

    actions = policy.sample(images, proprio, n_steps=3)
    assert actions.shape == (3, HORIZON, ACTION_DIM)
    assert torch.all(torch.isfinite(actions))


def test_agent_returns_one_action_per_call_and_replans(tmp_path):
    chunk_size = 2
    agent = make_agent(tmp_path, action_chunk_size=chunk_size)

    actions = [agent.get_action(make_observation()) for _ in range(chunk_size)]
    for action in actions:
        assert isinstance(action, np.ndarray)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action))

    # the whole chunk was consumed, so the next call has to sample a new one
    first_chunk = agent._action_chunk
    agent.get_action(make_observation())
    assert agent._action_chunk is not first_chunk
    assert agent._chunk_index == 1


def test_agent_caches_the_chunk_and_reset_clears_it(tmp_path):
    agent = make_agent(tmp_path, action_chunk_size=3)

    first_action = agent.get_action(make_observation())
    cached_chunk = agent._action_chunk
    second_action = agent.get_action(make_observation())
    # the second action comes from the cached chunk, not from a new sample
    assert agent._action_chunk is cached_chunk
    assert np.allclose(first_action, cached_chunk[0]) and np.allclose(second_action, cached_chunk[1])

    agent.reset()
    assert agent._action_chunk is None and agent._chunk_index == 0
    agent.get_action(make_observation())
    assert agent._action_chunk is not cached_chunk


def test_agent_defaults_to_executing_half_of_the_horizon(tmp_path):
    assert make_agent(tmp_path).action_chunk_size == HORIZON // 2

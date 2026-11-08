"""The normalization-stats pass keeps every batch it reads. Batches from DataLoader workers live in shared memory
that holds a file descriptor open while the tensor is alive, so keeping them uncopied crashed long datasets with
"RuntimeError: Too many open files" (86867 frames / batch 64 = 1358 batches > the default `ulimit -n` of 1024)."""

import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
resource = pytest.importorskip("resource")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from robot_imitation_glue.baseline import train, train_dit  # noqa: E402

FD_LIMIT = 128
N_BATCHES = 3 * FD_LIMIT


class ToyDataset(Dataset):
    def __init__(self, with_mask: bool):
        self.with_mask = with_mask

    def __len__(self):
        return N_BATCHES

    def __getitem__(self, idx):
        proprio = torch.full((3,), float(idx))
        if not self.with_mask:
            return torch.zeros(1), proprio, torch.full((2,), float(idx))
        actions = torch.full((4, 2), float(idx))
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
        return torch.zeros(1), proprio, actions, mask


@pytest.fixture
def low_fd_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (FD_LIMIT, hard))
    yield
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.skipif(sys.platform != "linux", reason="the file_descriptor sharing strategy is the linux default")
@pytest.mark.parametrize("module, with_mask", [(train, False), (train_dit, True)])
def test_stats_pass_does_not_leak_worker_file_descriptors(low_fd_limit, module, with_mask):
    # batch size 1 so every batch is its own shared-memory tensor
    loader = DataLoader(ToyDataset(with_mask), batch_size=1, num_workers=2)
    proprio_normalizer, action_normalizer = module.compute_normalization_stats(loader)

    expected_mean = (N_BATCHES - 1) / 2
    assert torch.allclose(proprio_normalizer.mean, torch.full((3,), expected_mean))
    assert torch.allclose(action_normalizer.mean, torch.full((2,), expected_mean))

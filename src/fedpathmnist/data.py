from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from medmnist import PathMNIST


@dataclass
class FederatedData:
    train_base: PathMNIST
    test_base: PathMNIST
    train_labels: np.ndarray
    test_labels: np.ndarray
    client_train_indices: list[np.ndarray]
    client_test_indices: list[np.ndarray]
    client_train_labels: list[np.ndarray]
    noise_counts: list[int]


class IndexedPathMNIST(Dataset):
    """An indexed PathMNIST view with optional client-specific train labels."""

    def __init__(
        self,
        base: PathMNIST,
        indices: Sequence[int],
        labels: Sequence[int] | None = None,
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(self.indices):
            raise ValueError("labels must have one entry per dataset index")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, original_label = self.base[int(self.indices[position])]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        label = (
            int(self.labels[position])
            if self.labels is not None
            else int(np.asarray(original_label).reshape(-1)[0])
        )
        return tensor, torch.tensor(label, dtype=torch.long)


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    num_classes: int,
) -> list[np.ndarray]:
    """Partition examples by drawing a client allocation per label class."""
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(class_indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            partitions[client_id].extend(class_indices[start : start + count].tolist())
            start += int(count)

    result = [np.asarray(part, dtype=np.int64) for part in partitions]
    for part in result:
        rng.shuffle(part)
    if any(len(part) == 0 for part in result):
        raise RuntimeError("Dirichlet split produced an empty client; try a larger alpha")
    return result


def _corrupt_labels(
    labels: np.ndarray,
    noise_rate: float,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Replace a fixed fraction of labels with uniformly sampled wrong classes."""
    if not 0.0 <= noise_rate < 1.0:
        raise ValueError("noise_rate must be in [0, 1)")
    noisy = labels.astype(np.int64, copy=True)
    count = int(round(len(noisy) * noise_rate))
    if count == 0:
        return noisy, 0

    selected = rng.choice(len(noisy), size=count, replace=False)
    offsets = rng.integers(1, num_classes, size=count)
    noisy[selected] = (noisy[selected] + offsets) % num_classes
    return noisy, count


def load_federated_pathmnist(
    data_dir: str | Path,
    num_clients: int = 5,
    alpha: float = 0.3,
    noisy_client: int = 0,
    noise_rate: float = 0.4,
    seed: int = 0,
) -> FederatedData:
    """Download PathMNIST and create reproducible train and client test splits."""
    if not 0 <= noisy_client < num_clients:
        raise ValueError("noisy_client must be a valid client index")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    root = str(data_path)
    train_base = PathMNIST(split="train", root=root, download=True, as_rgb=True)
    test_base = PathMNIST(split="test", root=root, download=True, as_rgb=True)
    train_labels = np.asarray(train_base.labels).reshape(-1).astype(np.int64)
    test_labels = np.asarray(test_base.labels).reshape(-1).astype(np.int64)
    num_classes = int(max(train_labels.max(), test_labels.max()) + 1)

    rng = np.random.default_rng(seed)
    client_train_indices = _dirichlet_partition(
        train_labels, num_clients, alpha, rng, num_classes
    )
    client_test_indices = _dirichlet_partition(
        test_labels, num_clients, alpha, rng, num_classes
    )

    client_train_labels: list[np.ndarray] = []
    noise_counts: list[int] = []
    for client_id, indices in enumerate(client_train_indices):
        labels = train_labels[indices]
        if client_id == noisy_client:
            labels, count = _corrupt_labels(labels, noise_rate, num_classes, rng)
        else:
            labels, count = labels.copy(), 0
        client_train_labels.append(labels)
        noise_counts.append(count)

    return FederatedData(
        train_base=train_base,
        test_base=test_base,
        train_labels=train_labels,
        test_labels=test_labels,
        client_train_indices=client_train_indices,
        client_test_indices=client_test_indices,
        client_train_labels=client_train_labels,
        noise_counts=noise_counts,
    )

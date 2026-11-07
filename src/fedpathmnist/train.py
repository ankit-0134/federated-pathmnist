from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from .data import IndexedPathMNIST, load_federated_pathmnist
from .model import SmallPathCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FedAvg or FedProx on PathMNIST")
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), required=True)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--noisy-client", type=int, default=0)
    parser.add_argument("--noise-rate", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--wandb-project", default="federated-pathmnist")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.algorithm == "fedprox" and args.mu < 0:
        parser.error("--mu must be non-negative")
    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("--rounds and --local-epochs must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_client(
    global_model: SmallPathCNN,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    round_index: int,
    client_id: int,
) -> tuple[dict[str, torch.Tensor], float, int, torch.Tensor]:
    local_model = copy.deepcopy(global_model).to(device)
    global_parameters = {
        name: parameter.detach().clone()
        for name, parameter in global_model.named_parameters()
    }
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        seed=args.seed + round_index * 1009 + client_id * 9176,
    )
    local_model.train()
    loss_sum = 0.0
    seen = 0

    for _ in range(args.local_epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(images)
            classification_loss = F.cross_entropy(logits, labels)
            objective = classification_loss
            if args.algorithm == "fedprox" and args.mu > 0:
                proximal_penalty = sum(
                    torch.sum((parameter - global_parameters[name]) ** 2)
                    for name, parameter in local_model.named_parameters()
                )
                objective = objective + (args.mu / 2.0) * proximal_penalty
            objective.backward()
            optimizer.step()
            batch_size = labels.size(0)
            loss_sum += float(classification_loss.detach()) * batch_size
            seen += batch_size

    delta_parts = []
    for name, parameter in local_model.named_parameters():
        delta_parts.append(
            (parameter.detach() - global_parameters[name]).reshape(-1).cpu()
        )
    delta = torch.cat(delta_parts)
    local_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }
    return local_state, loss_sum / max(seen, 1), seen, delta


@torch.no_grad()
def evaluate(model: SmallPathCNN, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.size(0)
    return loss_sum / max(count, 1), correct / max(count, 1)


def aggregate(
    client_states: list[dict[str, torch.Tensor]], sample_counts: list[int]
) -> dict[str, torch.Tensor]:
    total = float(sum(sample_counts))
    weights = [count / total for count in sample_counts]
    result: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        result[key] = sum(
            client_states[index][key] * weights[index]
            for index in range(len(client_states))
        )
    return result


def update_statistics(deltas: list[torch.Tensor]) -> tuple[list[float], list[float], float]:
    norms = [float(torch.linalg.vector_norm(delta)) for delta in deltas]
    matrix = torch.stack(deltas)
    normalized = F.normalize(matrix, p=2, dim=1, eps=1e-12)
    cosine = normalized @ normalized.T
    client_means = []
    for client_id in range(len(deltas)):
        peers = [index for index in range(len(deltas)) if index != client_id]
        client_means.append(float(cosine[client_id, peers].mean()))
    pairs = [float(cosine[i, j]) for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
    return norms, client_means, float(np.mean(pairs)) if pairs else 0.0


def write_outputs(
    run_dir: Path,
    rows: list[dict[str, Any]],
    model: SmallPathCNN,
    final_summary: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    torch.save(model.state_dict(), run_dir / "global_model.pt")

    rounds = [row["round"] for row in rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.plot(rounds, [row["global/test_accuracy"] for row in rows], linewidth=2.5, label="Global")
    client_ids = sorted(
        {
            int(key.split("/")[1].split("_")[1])
            for key in rows[0]
            if key.startswith("client/") and key.endswith("/test_accuracy")
        }
    )
    for client_id in client_ids:
        key = f"client/client_{client_id}/test_accuracy"
        axis.plot(rounds, [row[key] for row in rows], linewidth=1.2, alpha=0.8, label=f"Client {client_id}")
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("PathMNIST accuracy by communication round")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or (
        f"fedprox_mu{args.mu:g}_seed{args.seed}"
        if args.algorithm == "fedprox"
        else f"fedavg_seed{args.seed}"
    )

    experiment = load_federated_pathmnist(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        alpha=args.alpha,
        noisy_client=args.noisy_client,
        noise_rate=args.noise_rate,
        seed=args.seed,
    )
    train_datasets = [
        IndexedPathMNIST(experiment.train_base, indices, labels)
        for indices, labels in zip(
            experiment.client_train_indices, experiment.client_train_labels
        )
    ]
    test_datasets = [
        IndexedPathMNIST(experiment.test_base, indices)
        for indices in experiment.client_test_indices
    ]
    global_test = IndexedPathMNIST(
        experiment.test_base, np.arange(len(experiment.test_labels))
    )
    global_test_loader = make_loader(
        global_test, args.batch_size, False, args.num_workers, args.seed
    )
    local_test_loaders = [
        make_loader(dataset, args.batch_size, False, args.num_workers, args.seed + client_id)
        for client_id, dataset in enumerate(test_datasets)
    ]

    model = SmallPathCNN(num_classes=int(experiment.train_labels.max() + 1)).to(device)
    run_dir = args.output_dir / run_name
    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    config["device_used"] = str(device)
    config["client_train_sizes"] = [len(part) for part in experiment.client_train_indices]
    config["client_test_sizes"] = [len(part) for part in experiment.client_test_indices]
    config["client_noisy_label_counts"] = experiment.noise_counts

    run_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        mode=args.wandb_mode,
        dir=str(run_dir),
    )
    rows: list[dict[str, Any]] = []

    print(f"Run: {run_name} | device: {device} | rounds: {args.rounds}")
    print(f"Client train sizes: {config['client_train_sizes']}")
    print(f"Corrupted train labels: {experiment.noise_counts}")

    for round_index in range(1, args.rounds + 1):
        states = []
        losses = []
        sample_counts = []
        deltas = []
        for client_id, dataset in enumerate(train_datasets):
            state, loss, count, delta = train_client(
                model, dataset, args, device, round_index, client_id
            )
            states.append(state)
            losses.append(loss)
            sample_counts.append(count)
            deltas.append(delta)

        model.load_state_dict(aggregate(states, sample_counts))
        global_loss, global_accuracy = evaluate(model, global_test_loader, device)
        row: dict[str, Any] = {
            "round": round_index,
            "global/test_loss": global_loss,
            "global/test_accuracy": global_accuracy,
            "global/train_loss": float(np.average(losses, weights=sample_counts)),
        }
        norms, client_cosines, mean_cosine = update_statistics(deltas)
        row["updates/mean_pairwise_cosine"] = mean_cosine
        row["updates/mean_norm"] = float(np.mean(norms))

        for client_id, (loader, loss, count) in enumerate(
            zip(local_test_loaders, losses, sample_counts)
        ):
            client_loss, client_accuracy = evaluate(model, loader, device)
            prefix = f"client/client_{client_id}"
            row[f"{prefix}/test_loss"] = client_loss
            row[f"{prefix}/test_accuracy"] = client_accuracy
            row[f"{prefix}/train_loss"] = loss
            row[f"{prefix}/train_size"] = count
            row[f"updates/client_{client_id}_norm"] = norms[client_id]
            row[f"updates/client_{client_id}_mean_cosine"] = client_cosines[client_id]

        rows.append(row)
        run.log(row, step=round_index)
        print(
            f"round {round_index:03d}/{args.rounds} "
            f"global accuracy={global_accuracy:.4f} loss={global_loss:.4f} "
            f"mean update cosine={mean_cosine:.4f}"
        )

    final_summary = {
        "run_name": run_name,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "config": config,
        "final_round": rows[-1],
        "data_notes": {
            "client_train_sizes": config["client_train_sizes"],
            "client_test_sizes": config["client_test_sizes"],
            "client_noisy_label_counts": experiment.noise_counts,
            "noise_applied_to": args.noisy_client,
            "client_test_labels_are_clean": True,
        },
    }
    write_outputs(run_dir, rows, model, final_summary)
    run.summary.update(
        {
            "final/global_test_accuracy": rows[-1]["global/test_accuracy"],
            "final/global_test_loss": rows[-1]["global/test_loss"],
            "final/mean_pairwise_cosine": rows[-1]["updates/mean_pairwise_cosine"],
        }
    )
    run.finish()
    print(f"Saved metrics, plot, and checkpoint to {run_dir}")


if __name__ == "__main__":
    main()

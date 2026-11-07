# FedAvg and FedProx on PathMNIST

This project compares FedAvg and FedProx on the 9-class PathMNIST image-classification task. It uses 5 clients, a class-wise Dirichlet split with alpha 0.3, and fixed 40% symmetric label noise in client 0's training labels. The official PathMNIST test labels remain clean.

## Setup

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install dependencies:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

The first run downloads PathMNIST into the data directory. The script selects CUDA, Apple MPS, or CPU automatically; use `--device cpu` to force CPU.

## Run experiments

Set a W&B API key in your environment before online logging. Use one project for all runs:

~~~powershell
$env:WANDB_API_KEY = "your-api-key"
python -m src.fedpathmnist.train --algorithm fedavg --seed 0 --wandb-project federated-pathmnist
python -m src.fedpathmnist.train --algorithm fedprox --mu 0.01 --seed 0 --wandb-project federated-pathmnist
~~~

For an initial pipeline check, reduce the rounds with `--rounds 2`. The default is 20 communication rounds, one local epoch per client, batch size 128, and learning rate 0.01. These settings are a starting point and should be reported exactly as used. Add independent seeds by rerunning with `--seed 1` and `--seed 2`.

If W&B is not configured yet, use `--wandb-mode disabled` for a local run. The script still writes a CSV, JSON summary, accuracy plot, and final global checkpoint under `outputs/<run-name>/`. The two completed runs were saved offline. In this workspace, sync them with `wandb sync outputs/fedavg_seed0/wandb/offline-run-20261108_015130-u5hlakuu` and `wandb sync outputs/fedprox_mu0.01_seed0/wandb/offline-run-20261108_022519-48285hsv`. The project has not been published yet; authenticate with W&B, sync both runs, and make the project public before submitting its URL.

## Experiment design and metrics

- Every round, all five clients train from the current global weights and the server performs sample-count-weighted FedAvg aggregation.
- FedProx adds `mu / 2 * ||w_client - w_global||^2` to each local minibatch loss. The default `mu` is 0.01.
- Both methods use the same small CNN (three convolution layers with 32/64/128 channels, adaptive average pooling, and a 9-class linear head), SGD with momentum 0.9, learning rate 0.01, weight decay 0.0001, one local epoch, and batch size 128.
- For each class, training and test examples are independently allocated to clients using a Dirichlet draw with alpha 0.3. A fixed seed makes the allocation repeatable.
- Exactly 40% of client 0's training labels are replaced by uniformly sampled incorrect class labels. Evaluation labels are not changed.
- Global accuracy and loss use the complete official test set. Per-client accuracy and loss use clean, disjoint test subsets created by the same non-IID partitioning procedure.
- W&B and the local CSV include global/per-client train and test loss/accuracy, each client's update L2 norm, each client's mean cosine similarity to other client updates, and the mean pairwise cosine across clients.

Run names default to `fedavg_seed0` and `fedprox_mu0.01_seed0`. These runs were recorded in W&B offline mode because no W&B API key was configured. After setting `WANDB_API_KEY`, sync each generated offline run directory with `wandb sync <offline-run-directory>`, then set the project visibility to **Public** in W&B project settings before submitting its link. Do not publish API keys or credentials.

## Results

Both seed-0 runs completed for 20 rounds on CPU. The table reports the final communication round. These are single-seed results and should not be treated as evidence of statistical significance.

| Algorithm | Seed | Final global test accuracy | Final global test loss | W&B run |
|---|---:|---:|---:|---|
| FedAvg | 0 | 75.70% | 0.9681 | `fedavg_seed0` |
| FedProx (mu=0.01) | 0 | 76.42% | 0.9485 | `fedprox_mu0.01_seed0` |

| Client | Clean test samples | FedAvg accuracy | FedProx accuracy |
|---:|---:|---:|---:|
| 0 (40% train-label noise) | 1,020 | 72.06% | 72.84% |
| 1 | 460 | 85.00% | 85.65% |
| 2 | 516 | 71.32% | 67.64% |
| 3 | 1,291 | 77.23% | 78.16% |
| 4 | 3,893 | 75.62% | 76.86% |

FedProx is 0.72 percentage points higher in global accuracy in this run, while its client-2 accuracy is lower. The client-specific test partitions are disjoint but drawn independently from the training partitions. Read the per-client sample counts alongside accuracy because the test partitions are intentionally non-IID and differ in size. See the [global and per-client accuracy plot](deliverables/global_accuracy_comparison.png) and [final metrics CSV](deliverables/final_metrics.csv). Detailed round metrics are in each local `outputs/<run-name>/metrics.csv` file.

## Reading PDF and presentation

The one-page reading summary is available as [PDF](deliverables/reactive_optimizers_summary.pdf) and [Markdown](deliverables/reactive_optimizers_summary.md). The project overview is available as a [PowerPoint presentation](deliverables/federated_pathmnist_project.pptx). To regenerate the PDF and slides after changing the summary or experiment outputs:

~~~powershell
python -m pip install -r deliverables/requirements.txt
python deliverables/build_submission_files.py
~~~

The slides include a global accuracy comparison chart and a final per-client results table when both experiment outputs are available.

## Code changes and scope

This is a small standalone implementation of the requested baselines rather than a fork of PFLlib. It follows the same basic experiment structure while adding PathMNIST loading, Dirichlet client partitions, one noisy client, clean global and per-client evaluation, update-norm/cosine logging, CSV/JSON export, and an accuracy plot. This keeps the code path compact and makes the baseline implementations easy to inspect.

The experiment is a simulation: all clients run sequentially in one process and data are not transmitted over a network. Do not describe it as a deployed or privacy-preserving clinical system. MedMNIST is an educational benchmark and is not intended for clinical use.

## References

- [PFLlib](https://github.com/TsingZ0/PFLlib)
- [MedMNIST / PathMNIST](https://github.com/MedMNIST/MedMNIST)
- [Reactive Optimizers post](https://xingjianbai.com/reactive-optimizers/)

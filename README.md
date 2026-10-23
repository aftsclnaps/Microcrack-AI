# MicroCrack AI

Lightweight, difficulty-aware concrete crack detection (binary: **Crack / No Crack**) with
knowledge distillation and model compression. Research project; **not** a structural-safety tool.

## Status

| Stage | Content | State |
|---|---|---|
| 0/1 | Dataset inspection, duplicate-aware splits, MicroCrack-Tiny + torchvision baselines, training/eval | **implemented** |
| B | Ordinary + targeted augmentation experiments | planned |
| C | Difficulty annotation tool + per-category evaluation | planned |
| D | ResNet-18 teacher -> MicroCrack-Tiny distillation | planned |
| E | Quantization, pruning, resolution sweep, CPU benchmark | planned |
| F | Grad-CAM, uncertainty flag, Streamlit app, report | planned |

No result in this repository is claimed until it is produced by a run and stored in `artifacts/`.

## Setup

Local:
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # CPU-only torch: see pytorch.org for the right index URL
python -m pytest -q
```

Google Colab (GPU runtime):
```python
!git clone <your-repo-url> MicroCrack-AI && cd MicroCrack-AI && pip install -q -r requirements.txt
# Copy the dataset to the local Colab disk (reading 40k files straight from Drive is very slow):
!unzip -q /content/drive/MyDrive/concrete_crack.zip -d /content/data
%cd MicroCrack-AI
```
then use `--data-root /content/data/<folder containing Negative/ and Positive/>`.

## Stage 1 commands

```bash
# 1. Inspect, validate, group duplicates, split. Prints report; writes data/splits/*
python -m src.data.prepare_data --data-root /path/to/dataset

# 2. Smoke test (tiny subsample, 1 epoch; checks the plumbing only - ignore its metrics)
python -m src.training.train --data-root /path/to/dataset --max-samples 256 \
    --run-name smoke --set train.epochs=1 data.num_workers=0

# 3. Real baseline
python -m src.training.train --data-root /path/to/dataset --set model.name=microcrack_tiny

# 4. Other baselines (ImageNet-pretrained torchvision models need internet on first run)
python -m src.training.train --data-root /path/to/dataset --set model.name=mobilenet_v3_small model.pretrained=true train.lr=0.0005
python -m src.training.train --data-root /path/to/dataset --set model.name=shufflenet_v2_x1_0 model.pretrained=true train.lr=0.0005
python -m src.training.train --data-root /path/to/dataset --set model.name=resnet18 model.pretrained=true train.lr=0.0005

# 5. Re-evaluate any checkpoint
python -m src.evaluation.evaluate --checkpoint artifacts/runs/<run>/best.pt --split val
```

Lower learning rate for fine-tuned pretrained models is a starting guess, not a tuned value.
Choose settings with validation results only; the test split is evaluated once at the end of each run.

## Conventions
- Label 1 = Crack (positive class), 0 = No Crack. Model outputs 2 logits.
- `size_mb` = serialized `state_dict` in MiB (1024^2 bytes).
- Metrics that are undefined for a subset (e.g. recall with no positives) are `null`, never 0.
- Determinism: seeds are fixed and deterministic algorithms requested (`warn_only`), but bit-exact
  reproducibility across GPUs/driver/library versions is not guaranteed; AMP adds small variation.
- Freezing versions after your first successful run: `pip freeze > requirements.lock.txt`.

## Layout
```
configs/config.yaml        all hyper-parameters and paths
data/splits/               manifest.csv, split_info.json, dataset_report.json
src/data/                  prepare_data.py, splitting.py, dataset.py, transforms.py
src/models/                tiny_cnn.py, model_factory.py
src/training/              train.py, losses.py
src/evaluation/            metrics.py, evaluate.py
src/{optimization,explainability}/   reserved for later stages
tests/                     metrics, splitting, data-prep, model, dataset tests
artifacts/runs/<run>/      checkpoints, histories, metrics, predictions (git-ignored)
reports/figures/           generated figures
```

# beskar-test-3 — gpt-oss-20b Arabic fine-tuning

Fine-tunes OpenAI's open-source **gpt-oss-20b** model on the
[Arabic Instruct chatbot dataset](https://www.kaggle.com/datasets/omgits0mar/arabic-instruct-chatbot-dataset)
using [Unsloth](https://github.com/unslothai/unsloth) with 4-bit quantization and LoRA.
Originally a Kaggle notebook, converted to a plain Python script.

## Requirements

- Linux or WSL2 (Unsloth/bitsandbytes/Triton do not run natively on Windows)
- NVIDIA GPU with CUDA 12.x — ~14 GB VRAM (e.g. Kaggle T4/P100, RTX 3090/4090, A100)
- Python 3.10–3.12

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The dataset is downloaded automatically from Kaggle via `kagglehub`
(uses `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` if the dataset
requires authentication). To use a local copy instead:

```bash
DATA_PATH=/path/to/train.parquet python main.py
```

## Running on Beskar-Core

The repo ships an `orchestra.yml` pipeline for the Beskar-Core GPU worker.
The worker clones the repo, installs `requirements.txt`, and runs the steps;
everything written to `$OUTPUT_DIR` (`/scratch/workspace/artifacts`) is
collected as the job artifact. Tune the run via the `env` block:
`NUM_SAMPLES`, `MAX_STEPS`, `MODEL_NAME`.

Local dry-run of the pipeline:

```bash
python run_pipeline.py orchestra.yml --dry-run   # from Beskar-Core/spawner/worker
```

## Outputs

All outputs are written under `$OUTPUT_DIR` (defaults to the repo root locally):

- `outputs/` — training checkpoints
- `gpt-oss-arabic-finetuned-v1/` — final LoRA adapters
- `log_history.pt` — training log history
- `training_loss.png` — loss curve
- A sample Arabic generation is printed at the end of the run

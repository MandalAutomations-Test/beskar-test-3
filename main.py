
import warnings

# Suppress deprecation noise from torch internals before any heavy imports.
warnings.filterwarnings(
    "ignore",
    message="torch._dynamo.config.inline_inbuilt_nn_modules is deprecated",
    category=FutureWarning,
)

# Unsloth must be imported before transformers/trl so its patches apply.
from unsloth import FastLanguageModel

import glob
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-0.8B-GGUF")
KAGGLE_DATASET = "omgits0mar/arabic-instruct-chatbot-dataset"
MAX_SEQ_LENGTH = 1080
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "5000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))

# All outputs go under OUTPUT_DIR so runners like Beskar-Core can collect them
# as a single artifact folder; defaults to the current directory for local runs.
ARTIFACT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
CHECKPOINT_DIR = ARTIFACT_DIR / "outputs"
FINAL_MODEL_DIR = ARTIFACT_DIR / "Qwen3.5-0.8B-GGUF-finetuned-v1"
LOG_HISTORY_PATH = ARTIFACT_DIR / "log_history.pt"
LOSS_PLOT_PATH = ARTIFACT_DIR / "training_loss.png"


def find_dataset_parquet() -> str:
    """Locate the training parquet: $DATA_PATH, the Kaggle input dir, or a kagglehub download."""
    data_path = os.environ.get("DATA_PATH")
    if data_path:
        return data_path

    kaggle_glob = glob.glob("/kaggle/input/arabic-instruct-chatbot-dataset/*.parquet")
    if kaggle_glob:
        return kaggle_glob[0]

    import kagglehub

    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    parquets = glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True)
    if not parquets:
        raise FileNotFoundError(f"No parquet file found in {dataset_dir}")
    return parquets[0]


_BYTES_PER_GB = 1024 ** 3
# Overhead factor for VRAM estimation: raw 4-bit weight storage (0.5) plus
# approximately 0.1 bytes/param for LoRA activations, KV-cache, and optimizer
# state that lives on the GPU even with 4-bit quantization.
_VRAM_OVERHEAD_BYTES_PER_PARAM = 0.6
# Rough map of common model sizes (billions of parameters) from the model name.
_MODEL_SIZE_HINTS = {
    "0.5b": 0.5, "1b": 1, "1.5b": 1.5, "3b": 3, "4b": 4, "7b": 7,
    "8b": 8, "9b": 9, "11b": 11, "13b": 13, "14b": 14, "20b": 20,
    "32b": 32, "70b": 70,
}


def _estimate_model_params_b(model_name: str) -> float | None:
    """Return estimated parameter count in billions from the model name, or None."""
    lower = model_name.lower()
    for suffix, size in _MODEL_SIZE_HINTS.items():
        if suffix in lower:
            return size
    return None


def _check_vram(model_name: str) -> None:
    """Warn if the available VRAM is likely too small for the requested model."""
    if not torch.cuda.is_available():
        return
    params_b = _estimate_model_params_b(model_name)
    if params_b is None:
        return
    free_gb = round(torch.cuda.mem_get_info()[0] / _BYTES_PER_GB, 2)
    total_gb = round(torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB, 2)
    # Minimum GPU memory: 4-bit weights + LoRA activations + optimizer buffers.
    min_needed_gb = round(params_b * 1e9 * _VRAM_OVERHEAD_BYTES_PER_PARAM / _BYTES_PER_GB, 1)
    print(
        f"VRAM check: {model_name!r} ~{params_b}B params -> "
        f"needs >={min_needed_gb} GB; GPU has {free_gb}/{total_gb} GB free."
    )
    if free_gb < min_needed_gb:
        print(
            f"WARNING: The model likely won't fit in GPU memory "
            f"({free_gb} GB free < {min_needed_gb} GB needed).\n"
            "  * Use a smaller model (e.g. unsloth/gpt-oss-8b) or a GPU with more VRAM.\n"
            "  * The script will continue and attempt CPU offloading, but training "
            "will be very slow."
        )


def load_model():
    _check_vram(MODEL_NAME)
    print(f"Loading model {MODEL_NAME!r} -- this may take several minutes on first run "
          "(weights are downloaded and kernels are compiled for your GPU)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=None,  # None for auto detection
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # 4-bit quantization to reduce memory
        full_finetuning=False,
    )
    print("Model loaded successfully.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # 0 is optimized in Unsloth
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def prepare_dataset(tokenizer):
    parquet_path = find_dataset_parquet()
    print(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    dataset = Dataset.from_pandas(df[["instruction", "output"]][:NUM_SAMPLES])

    def format_arabic_dataset(examples):
        chats = [
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            for instruction, output in zip(examples["instruction"], examples["output"])
        ]
        texts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in chats
        ]
        return {"text": texts}

    formatted = dataset.map(format_arabic_dataset, batched=True)
    print("Sample formatted example:")
    print(formatted[0]["text"])
    return formatted


def train(model, tokenizer, formatted_dataset):
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=256,  # single max length for the whole conversation
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=MAX_STEPS,  # set num_train_epochs=1 instead for a full run
            learning_rate=2e-5,
            max_grad_norm=0.3,
            logging_steps=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(CHECKPOINT_DIR),
            report_to="none",
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    torch.save(trainer.state.log_history, LOG_HISTORY_PATH)
    print("Training finished and artifacts saved.")

    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {round(used_memory / max_memory * 100, 3)} %.")
    print(f"Peak reserved memory for training % of max memory = {round(used_for_lora / max_memory * 100, 3)} %.")


def plot_loss():
    import matplotlib

    matplotlib.use("Agg")  # no display needed when running as a script
    import matplotlib.pyplot as plt

    log_history = torch.load(LOG_HISTORY_PATH)
    log_df = pd.DataFrame(log_history)
    loss_df = log_df[log_df["loss"].notna()].copy()

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["step"], loss_df["loss"], marker="o", linestyle="-", markersize=4)
    plt.title("Training Loss Over Steps", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.savefig(LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to {LOSS_PLOT_PATH}")


def run_inference(model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "reasoning language: Arabic\n\nYou are a helpful assistant. اجب علي الاتي بالعربي فقط.",
        },
        {"role": "user", "content": "ما هي طريقة عمل البيتزا , اجب في خطوات"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",  # medium, high
    ).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=1024, streamer=TextStreamer(tokenizer)
    )
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("\n--- Generated response ---")
    print(generated_text)


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. This script requires a Linux machine "
            "(or WSL2) with an NVIDIA GPU."
        )
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Capability:", torch.cuda.get_device_capability())
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()
    formatted_dataset = prepare_dataset(tokenizer)
    train(model, tokenizer, formatted_dataset)
    plot_loss()
    run_inference(model, tokenizer)


if __name__ == "__main__":
    main()

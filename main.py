# %% [markdown]
# ## <b><span style='color:#9146ff'>|</span> Introduction </b>
# 
# Welcome to this notebook on fine-tuning **OpenAI's new open-source model, GPT-OSS**, on an Arabic instruct dataset. We'll be using Unsloth's high-performance kernels to make the process incredibly fast and memory-efficient on free Kaggle resources! 🎉
# 
# In this notebook, you will learn how to:
# 
# * Set up the environment and install necessary dependencies.
# * Prepare and preprocess the Arabic dataset for model training.
# * Fine-tune the powerful `GPT-OSS` model for a specific task.
# * Leverage **Unsloth** for 4-bit quantization and 2x faster training.
# * Use Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
# * Utilize the SFT Trainer for fine-tuning.
# * Choose appropriate hyperparameters for training.
# * Test the performance of the fine-tuned model.
# 
# Note : You can generalize this notebook on any other different QA instruct dataset for chatbot

# %% [markdown]
# ![gpt-oss.png](attachment:7df92dfe-129a-45a2-b0af-314b8301a512.png)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T07:34:43.88453Z","iopub.execute_input":"2025-08-11T07:34:43.884819Z","iopub.status.idle":"2025-08-11T07:34:43.888614Z","shell.execute_reply.started":"2025-08-11T07:34:43.884794Z","shell.execute_reply":"2025-08-11T07:34:43.887989Z"}}
# from huggingface_hub import login
# from kaggle_secrets import UserSecretsClient

# secret_label = "HF Hub"
# secret_value = UserSecretsClient().get_secret(secret_label)
# login(token=secret_value)

# %% [markdown]
# ## <b>1 <span style='color:#9146ff'>|</span> Instalation and Logging </b>

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:29:23.961453Z","iopub.execute_input":"2025-08-11T09:29:23.961732Z","iopub.status.idle":"2025-08-11T09:31:56.89253Z","shell.execute_reply.started":"2025-08-11T09:29:23.961709Z","shell.execute_reply":"2025-08-11T09:31:56.891511Z"}}
%%capture
# We're installing the latest Torch, Triton, OpenAI's Triton kernels, Transformers and Unsloth!
!pip install --upgrade -qqq uv
try: import numpy; install_numpy = f"numpy=={numpy.__version__}"
except: install_numpy = "numpy"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {install_numpy} \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    torchvision bitsandbytes \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels


# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:31:56.894142Z","iopub.execute_input":"2025-08-11T09:31:56.894443Z","iopub.status.idle":"2025-08-11T09:31:59.898977Z","shell.execute_reply.started":"2025-08-11T09:31:56.894409Z","shell.execute_reply":"2025-08-11T09:31:59.898342Z"}}
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Capability:", torch.cuda.get_device_capability())


# %% [markdown]
# ## <b>2 <span style='color:#9146ff'>|</span> Model Configuration and Quantization </b>
# 
# We'll use Unsloth's `FastLanguageModel` to load our model. This is a key step that simplifies the entire process. `FastLanguageModel` automatically handles 4-bit quantization and applies significant speed optimizations, abstracting away the complexities of `BitsAndBytesConfig` and manual setup.
# 
# After loading the base model, we'll use `get_peft_model` to inject LoRA adapters into it. This prepares the model for Parameter-Efficient Fine-Tuning, where we only train a tiny fraction of the total parameters, saving immense amounts of VRAM.
# 
# **Key Components in the Code :**
# 
# - `FastLanguageModel.from_pretrained`: The core Unsloth function for loading models.
# 
# > `model_name`: We're using a pre-quantized GPT-OSS model from the Unsloth Hub, which allows for faster downloads and no out-of-memory errors during loading.
# 
# > `load_in_4bit = True`: This single flag tells Unsloth to load the model with 4-bit quantization, dramatically reducing its memory footprint.
# 
# - `FastLanguageModel.get_peft_model`: This function configures the LoRA adapters.
# 
# > `r`: The rank of the LoRA matrices. This determines the number of trainable parameters. A higher rank means more expressive power but also more memory usage. `8` or `16` is a great starting point.
# 
# > `target_modules`: A list of the model layers (like the attention projections `q_proj`, `v_proj`, etc.) where we will inject the trainable LoRA adapters. Unsloth automatically finds these for you if not specified.
# 
# > `lora_alpha`: A scaling factor for the LoRA weights. A common convention is to set this to 2 * r.
# 
# > `bias = "none"`: An Unsloth-specific optimization that further reduces trainable parameters.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:31:59.942494Z","iopub.execute_input":"2025-08-11T09:31:59.942911Z","iopub.status.idle":"2025-08-11T09:33:47.871759Z","shell.execute_reply.started":"2025-08-11T09:31:59.942893Z","shell.execute_reply":"2025-08-11T09:33:47.871174Z"}}
from unsloth import FastLanguageModel
import torch

max_seq_length = 1080
dtype = None

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b", 
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# %% [markdown]
# ### LoRA (Low-Rank Adaptation) :
# is a technique for Parameter-Efficient Fine-Tuning (PEFT) that adds trainable low-rank matrices to the model weights.
# 
# ![LoRa](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/133_trl_peft/step2.png)
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:47.87247Z","iopub.execute_input":"2025-08-11T09:33:47.872651Z","iopub.status.idle":"2025-08-11T09:33:53.005409Z","shell.execute_reply.started":"2025-08-11T09:33:47.872636Z","shell.execute_reply":"2025-08-11T09:33:53.004607Z"}}
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# %% [markdown]
# ## <b>3 <span style='color:#9146ff'>|</span> Data Preparation </b>
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:53.006266Z","iopub.execute_input":"2025-08-11T09:33:53.006494Z","iopub.status.idle":"2025-08-11T09:33:53.499013Z","shell.execute_reply.started":"2025-08-11T09:33:53.006475Z","shell.execute_reply":"2025-08-11T09:33:53.498355Z"}}
import pandas as pd

# Load the Parquet file
df = pd.read_parquet('/kaggle/input/arabic-instruct-chatbot-dataset/train-00000-of-00001-10520e8228c2c104.parquet')

# Display the first few rows
df.head()


# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:53.549517Z","iopub.execute_input":"2025-08-11T09:33:53.549709Z","iopub.status.idle":"2025-08-11T09:33:53.601669Z","shell.execute_reply.started":"2025-08-11T09:33:53.549693Z","shell.execute_reply":"2025-08-11T09:33:53.601127Z"}}
from datasets import Dataset
# We only need the instruction and output columns
dataset = Dataset.from_pandas(df[['instruction', 'output']][:5000])

# %% [markdown]
# ## <b>4 <span style='color:#9146ff'>|</span> Data Preprocessing </b>
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:53.602321Z","iopub.execute_input":"2025-08-11T09:33:53.602496Z","iopub.status.idle":"2025-08-11T09:33:53.607223Z","shell.execute_reply.started":"2025-08-11T09:33:53.602482Z","shell.execute_reply":"2025-08-11T09:33:53.6066Z"}}
# This function converts each row into the required chat format
def format_arabic_dataset(examples):
    # The chat template requires a list of dictionaries
    # with 'role' and 'content' keys
    chats = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        chat = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        chats.append(chat)

    # Apply the template and return a new 'text' column
    # SFTTrainer will handle the tokenization automatically
    texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]
    return {"text": texts}


# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:53.636984Z","iopub.execute_input":"2025-08-11T09:33:53.637185Z","iopub.status.idle":"2025-08-11T09:33:54.498673Z","shell.execute_reply.started":"2025-08-11T09:33:53.637171Z","shell.execute_reply":"2025-08-11T09:33:54.498074Z"}}
formatted_dataset = dataset.map(format_arabic_dataset, batched=True)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:54.499413Z","iopub.execute_input":"2025-08-11T09:33:54.499593Z","iopub.status.idle":"2025-08-11T09:33:54.503982Z","shell.execute_reply.started":"2025-08-11T09:33:54.499579Z","shell.execute_reply":"2025-08-11T09:33:54.503201Z"}}
print(formatted_dataset[0]['text'])

# %% [markdown]
# ## <b>5 <span style='color:#9146ff'>|</span> Model Training and Fine-tuning </b>
# 

# %% [markdown]
# ### SFTTrainer:
# 
# - Supervised Fine-tuning (SFT): Optimized for fine-tuning pre-trained models with smaller datasets on supervised learning tasks.
# - Simpler interface: Provides a streamlined workflow with fewer configuration options, making it easier to get started.
# - Efficient memory usage: Uses techniques like parameter-efficient (PEFT) and packing optimizations to reduce memory consumption during training.
# - Faster training: Achieves comparable or better accuracy with smaller datasets and shorter training times than Trainer.

# %% [markdown]
# ### Training Arguments :
# 
# **Parameter Explanations**
# 
# **Batching & Training Duration**
# 
# > `per_device_train_batch_size`: The number of samples processed on each GPU per step. This is a key lever for managing VRAM.
# 
# > `gradient_accumulation_steps`: Number of steps to accumulate gradients before performing a model update. The effective batch size is `batch_size * num_gpus * accumulation_steps`.
# 
# > `max_steps` or `num_train_epochs`: You can specify either the total number of training steps or the number of full passes over the dataset (epochs).
# 
# **Optimizer & Learning Rate**
# 
# > `optim`: The optimizer to use. Unsloth is highly optimized for `adamw_8bit`.
# 
# > `learning_rate`: The speed at which the model learns. A smaller value like `2e-5` is a safe starting point for fine-tuning.
# 
# > `lr_scheduler_type`: The learning rate schedule. `"cosine"` is a popular choice that gradually decreases the learning rate, helping the model settle into a good minimum.
# 
# > `warmup_ratio`: The fraction of training steps used to warm up the learning rate from 0 to its target value. This helps stabilize training at the beginning.
# 
# **Performance & Memory Optimization**
# > `fp16` or `bf16`: Enables mixed-precision training. Use `fp16` for NVIDIA T4/P100/V100 GPUs and `bf16` for Ampere (A100) or newer GPUs for better stability.
# 
# > `max_grad_norm`: Gradient clipping. This acts as a safety rail to prevent exploding gradients, which can cause NaN loss and training instability.
# 
# > `group_by_length`: Groups samples of similar length into batches. This reduces the amount of padding needed and can significantly speed up training.
# 
# **Logging & Saving**
# > `output_dir`: The directory where model checkpoints and final adapters will be saved.
# 
# > `logging_steps`: How often to print training metrics (like loss) to the console.
# 
# > `save_steps`: How often to save a model checkpoint.

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:54.504701Z","iopub.execute_input":"2025-08-11T09:33:54.504882Z","iopub.status.idle":"2025-08-11T09:33:59.234757Z","shell.execute_reply.started":"2025-08-11T09:33:54.504867Z","shell.execute_reply":"2025-08-11T09:33:59.233805Z"}}
from trl import SFTConfig, SFTTrainer

tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field="text",      # The column we just created
    max_seq_length=256,             # A single max length for the whole conversation
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 50,
        learning_rate = 2e-5,      # Reduced learning rate
        max_grad_norm = 0.3,       # Added gradient clipping
        logging_steps = 2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:59.23591Z","iopub.execute_input":"2025-08-11T09:33:59.236246Z","iopub.status.idle":"2025-08-11T09:33:59.241941Z","shell.execute_reply.started":"2025-08-11T09:33:59.236217Z","shell.execute_reply":"2025-08-11T09:33:59.241201Z"}}
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:33:59.242808Z","iopub.execute_input":"2025-08-11T09:33:59.243171Z","iopub.status.idle":"2025-08-11T09:56:52.173407Z","shell.execute_reply.started":"2025-08-11T09:33:59.243147Z","shell.execute_reply":"2025-08-11T09:56:52.172638Z"}}
trainer_stats = trainer.train()

# %% [markdown]
# ### Optional: for clearing GPU Vram mem from cache

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T08:58:40.917333Z","iopub.execute_input":"2025-08-11T08:58:40.917612Z","iopub.status.idle":"2025-08-11T08:58:42.1839Z","shell.execute_reply.started":"2025-08-11T08:58:40.917594Z","shell.execute_reply":"2025-08-11T08:58:42.183302Z"}}
# import torch
# import gc

# # Run garbage collection
# gc.collect()

# # Clear PyTorch CUDA cache
# torch.cuda.empty_cache()

# # Optional: Reset memory stats if you want to monitor from zero
# # torch.cuda.reset_peak_memory_stats()
# print("GPU memory cleared.")

# %% [markdown]
# ### Save model & Publish

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T10:11:37.281343Z","iopub.execute_input":"2025-08-11T10:11:37.281672Z","iopub.status.idle":"2025-08-11T10:11:38.319224Z","shell.execute_reply.started":"2025-08-11T10:11:37.28165Z","shell.execute_reply":"2025-08-11T10:11:38.318419Z"}}
# Save the final model and training history
trainer.save_model("gpt-oss-arabic-finetuned-v1")
torch.save(trainer.state.log_history, "log_history.pt")
print("Training finished and artifacts saved.")

# Optional: For uploading your model to HuggingFace
# ------------------------------------------------
# Define your Hugging Face repo ID
# hf_repo_id = "YourUsername/YourArabicChatbotModelName" # Replace with your info

# # Push to the hub
# model.push_to_hub(hf_repo_id, token = True)
# tokenizer.push_to_hub(hf_repo_id, token = True)

# print(f"Model successfully pushed to https://huggingface.co/{hf_repo_id}")

# %% [markdown]
# ## <b>6 <span style='color:#9146ff'>|</span> Model Evaluation and Vizualization </b>
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T10:11:58.385763Z","iopub.execute_input":"2025-08-11T10:11:58.386086Z","iopub.status.idle":"2025-08-11T10:11:58.665939Z","shell.execute_reply.started":"2025-08-11T10:11:58.386063Z","shell.execute_reply":"2025-08-11T10:11:58.665149Z"}}
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved log history
log_history = torch.load("log_history.pt")

# Convert to a pandas DataFrame
log_df = pd.DataFrame(log_history)

# Filter for rows that contain loss values and drop any that don't
loss_df = log_df[log_df['loss'].notna()].copy()

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10, 6))
plt.plot(loss_df['step'], loss_df['loss'], marker='o', linestyle='-', markersize=4)
plt.title('Training Loss Over Steps', fontsize=16)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T10:13:48.509346Z","iopub.execute_input":"2025-08-11T10:13:48.509671Z","iopub.status.idle":"2025-08-11T10:13:48.516135Z","shell.execute_reply.started":"2025-08-11T10:13:48.50965Z","shell.execute_reply":"2025-08-11T10:13:48.515362Z"}}
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown]
# ## <b>7 <span style='color:#9146ff'>|</span> Testing the model performance on a single inference </b>
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T10:17:09.416986Z","iopub.execute_input":"2025-08-11T10:17:09.417326Z","iopub.status.idle":"2025-08-11T10:19:39.920324Z","shell.execute_reply.started":"2025-08-11T10:17:09.417303Z","shell.execute_reply":"2025-08-11T10:19:39.919756Z"}}
messages = [
    {"role": "system", "content": "reasoning language: Arabic\n\nYou are a helpful assistant. اجب علي الاتي بالعربي فقط."},
    {"role": "user", "content": "ما هي طريقة عمل البيتزا , اجب في خطوات"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "low", #medium, high
).to(model.device)
from transformers import TextStreamer
_ = model.generate(**inputs, max_new_tokens = 1024, streamer = TextStreamer(tokenizer))

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T10:24:23.119408Z","iopub.execute_input":"2025-08-11T10:24:23.119689Z","iopub.status.idle":"2025-08-11T10:24:23.126535Z","shell.execute_reply.started":"2025-08-11T10:24:23.119667Z","shell.execute_reply":"2025-08-11T10:24:23.125839Z"}}
from IPython.display import display, Markdown

generated_tokens = _[0, inputs['input_ids'].shape[1]:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# 4. Display the captured text as formatted Markdown
display(Markdown(generated_text))

# %% [code]

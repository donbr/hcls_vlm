# %% [markdown]
# <a href="https://colab.research.google.com/gist/donbcolab/014b0deb8dd76c90fdfe481b33181890/paligemma_rlaif_fine_tuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Training PaliGemma with RLAIF-V Dataset
# 
# This notebook demonstrates the process of fine-tuning the PaliGemma model using the RLAIF-V dataset. The steps include loading the dataset, preprocessing, setting up the model and training configurations, handling potential GPU memory issues, and resuming training from checkpoints.
# 
# **Contents:**
# 1. Setup and Imports
# 2. Load and Preprocess Dataset
# 3. Configure Model and Training
# 4. Manage GPU Memory and Checkpoints
# 5. Training and Evaluation
# 6. Pushing Model to Hugging Face Hub
# 

# %%
!pip install -q -U git+https://github.com/huggingface/transformers.git datasets accelerate bitsandbytes peft

# %%
from huggingface_hub import notebook_login
notebook_login()

# %% [markdown]
# ## 1. Setup and Imports
# This section covers the installation and import of necessary libraries and dependencies.

# %%
# Import the required libraries
from datasets import load_dataset
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig

# %% [markdown]
# ## 2. Load and Preprocess Dataset
# In this section, we load the RLAIF-V dataset and preprocess it by removing unnecessary columns and splitting it into training and validation subsets.

# %%
try:
    # Try to load the dataset normally
    ds = load_dataset('openbmb/RLAIF-V-Dataset', split="train[:1%]")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Load the dataset again with verifications ignored
    ds = load_dataset('openbmb/RLAIF-V-Dataset', verification_mode='no_checks')


# %% [markdown]
# Block 3: Verify Dataset Size
# Print the number of records in the train split to compare with the expected count.

# %%
# Print the number of records in the train split
print(f"Number of examples in train split: {len(ds)}")


# %% [markdown]
# Block 4: Remove Unnecessary Columns
# After verifying the dataset, remove unnecessary columns.

# %%
# Columns to remove
cols_remove = ["ds_name", "origin_dataset", "rejected", "origin_split", "idx", "image_path"]

# Remove the columns
ds = ds.remove_columns(cols_remove)


# %%
ds

# %% [markdown]
# Block 5: Split Dataset
# Split the dataset into training and validation subsets.

# %%
# Split the dataset into train and validation subsets
train_val_split = ds.train_test_split(test_size=0.1)
train_ds = train_val_split['train']
val_ds = train_val_split['test']


# %% [markdown]
# Block 6: Further Subset
# Take 1% of the train and validation subsets.

# %%
# Take 1% of the train subset
# train_ds_split = train_ds_init.train_test_split(test_size=0.01, stratify_by_column=None)
# train_ds = train_ds_split['test']

# # Take 1% of the validation subset
# val_ds_split = val_ds_init.train_test_split(test_size=0.01, stratify_by_column=None)
# val_ds = val_ds_split['test']


# %% [markdown]
# Block 7: Verify Splits
# Finally, print the sizes of the resulting datasets to verify.

# %%
# Check the sizes of the splits
print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")


# %% [markdown]
# ## 3. Configure Model and Training
# We set up the PaliGemma model, configure the training arguments, and initialize the Trainer.

# %%
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example['chosen'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]

    tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)

    return tokens


# %%
# Alternatively, load the model in 4-bit for QLoRA
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                          quantization_config=bnb_config,
                                                          device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
print(f"Model Architecture:  {model}")

# %% [markdown]
# Step 9: Set Up Training Arguments and Trainer
# Configure training arguments and set up the Trainer.

# %%
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

args = TrainingArguments(
    num_train_epochs=8,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    output_dir="paligemma_rlaifv-V-1",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    args=args
)



# %% [markdown]
# Step 10: Start Training
# Train the model.

# %%
trainer.train()

# trainer.train(resume_from_checkpoint=last_checkpoint)

# %% [markdown]
# Step 11: Push Model to Hub
# Push the trained model and training artifacts to Hugging Face Hub.

# %%
trainer.push_to_hub()

# %% [markdown]
# ## Target Modules for PeFT using LoRA
# 
# Transformer model parameter modules targeted for PeFT (parameter-efficient fine-tuning) using LoRA (Low-Rank Adaptation).
# - q_proj
# - o_proj
# - k_proj
# - v_proj
# - gate_proj
# - up_proj
# - down_proj
# 
# ### Explanation of Target Modules
# 
# 1. **q_proj (Query Projection)**:
#    - This module is responsible for transforming the input tokens into query vectors in the self-attention mechanism. Queries are used to calculate attention scores, determining how much focus to give to different parts of the input sequence.
# 
# 2. **k_proj (Key Projection)**:
#    - This module transforms input tokens into key vectors in the self-attention mechanism. Keys, along with queries, are used to compute the attention scores.
# 
# 3. **v_proj (Value Projection)**:
#    - This module transforms input tokens into value vectors in the self-attention mechanism. Values are the actual data that is combined based on the attention scores to produce the output of the attention mechanism.
# 
# 4. **o_proj (Output Projection)**:
#    - This module transforms the output of the attention mechanism into the final representation used for further processing in the model. It combines the attended values from the attention mechanism into a single output.
# 
# 5. **gate_proj (Gating Projection)**:
#    - This module often refers to mechanisms that control the flow of information within the model. Gating mechanisms can be used to dynamically adjust the contribution of different parts of the model based on the input data.
# 
# 6. **up_proj (Up Projection)**:
#    - This module typically refers to a projection that increases the dimensionality of the data. It's used in various places in the model, including in feed-forward networks or other transformation layers.
# 
# 7. **down_proj (Down Projection)**:
#    - This module typically refers to a projection that decreases the dimensionality of the data. It can be used to reduce the size of representations after processing, helping to manage the model’s complexity and computational requirements.
# 
# ### Purpose in LoRA
# 
# In the context of LoRA, targeting these specific modules means that the low-rank adaptation will be applied only to these components of the model. This selective fine-tuning allows for efficient training by focusing on the most impactful parts of the model, thus significantly reducing the number of trainable parameters while maintaining performance.
# 
# ### Practical Benefits
# 
# 1. **Efficiency**:
#    - By focusing on key projections and transformations, LoRA can achieve significant reductions in computational cost and memory usage.
# 
# 2. **Performance**:
#    - Targeting essential components like query, key, value, and output projections ensures that the fine-tuning process directly impacts the model's core capabilities, preserving or even enhancing performance.
# 
# 3. **Scalability**:
#    - This approach is particularly beneficial when dealing with very large models, where full fine-tuning would be impractical due to resource constraints.
# 
# ### References
# 
# - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): This paper introduces the concept of LoRA and explains the targeted fine-tuning approach in detail.
# - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index): The official documentation provides insights into various model components and their roles in transformer architectures.

# %%




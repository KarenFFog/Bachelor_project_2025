import sys
import os
from generate import *

# ===== CONFIGURATION =====
root = "/home/fhd511/Geollm_project/BigEarthNet_data_train_s2/BigEarthNet-v1.0"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
access_token = os.getenv("HF_TOKEN")
precision = 3
batch_size = 64

# Prompt to generate descriptions
prompt_template = ("Give a description of the general geographical setting at {coords}, paying attention to terrain, climate, flora and fauna, and other distinctive natural features.")


# ===== SPLIT ARGUMENT =====
split = sys.argv[1]  # e.g., "aa", "ab", etc.

split_file = os.path.join(root, f"locations_part_{split}")
metadata_file = f"metadata_part_{split}.jsonl"
output_file = f"descriptions_part_{split}.jsonl"

# ===== LOAD FOLDER LIST =====
with open(split_file, "r") as f:
    folder_list = [os.path.basename(line.strip()) for line in f if line.strip()]


# ===== CREATE DATASET AND GENERATE METADATA =====
#dataset = BigEarthNetS2Custom(root, folder_list=folder_list)
#save_samples_to_jsonl(dataset, metadata_file)

# ===== RUN GENERATION =====
run_single_prompt(
    prompt_template=prompt_template,
    data_path=metadata_file,
    n_loc=len(folder_list),
    model_name=model_name,
    token=access_token,
    output_file=output_file,
    precision=precision,
    batch_size=batch_size
)

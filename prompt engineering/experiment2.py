# After experiment 1, create embeddings of generated response and find best matching land cover labels

import sys
import os
import argparse
from gen import *

# Argument parser
parser = argparse.ArgumentParser(description="Run experiment 2 with configurable settings.")
parser.add_argument('--per_sentence', type=str, required=True, choices=["true", "false"],
                    help="Whether to run per sentence (true/false)")
parser.add_argument('--mode', type=str, required=True, choices=["whole", "sentence"],
                    help="Output mode: 'whole' for whole text or 'sentence' for sentence-by-sentence")

args = parser.parse_args()

# Convert argument string to boolean
per_sentence_flag = args.per_sentence.lower() == "true"
output_mode = "whole text" if args.mode == "whole" else "sentence-by-sentence"

precisions = list(range(8))
#prompt_ids = list(range(1, 13))
prompt_ids = [1]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Geollm_project

for prec in precisions:
    for prompt_id in prompt_ids:
        input_path = os.path.join(BASE_DIR, f"prompt engineering/results/experiment_1_extra/precision_{prec}/results_prompt_{prompt_id}.jsonl")
        output_path = os.path.join(BASE_DIR, f"prompt engineering/results/experiment_2_extra/{output_mode}/precision_{prec}/predicted_labels_{prompt_id}.jsonl")
        emb_l_path = os.path.join(BASE_DIR, "Data/labels_with_embeddings.jsonl")

        process_and_label_sentences(input_path, emb_l_path, output_path, per_sentence=per_sentence_flag)


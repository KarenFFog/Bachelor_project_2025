# After experiment 1, create embeddings of generated response and find best matching land cover labels

import sys
# whole_text or sentence_by_sentence

precisions = [0, 1, 2, 3, 4, 5, 6, 7]
prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for prec in precisions:
    for prompt_id in prompt_ids:
        input_path = f"prompt engineering/results/experiment_1/precision_{prec}/results_prompt_{prompt_id}.jsonl"
        output_path = f"prompt engineering/results/experiment_2/whole text/precision_{prec}/predicted_labels_{prompt_id}.jsonl"
        emb_l_path = "Data/labels_with_embeddings.jsonl"

        process_and_label_sentences(input_path, emb_l_path, output_path, per_sentence=False)


# Generate describtions for n_loc locations, using different prompt and different precisions.

from eval import *
from gen import *
from plot_func import *


# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
data_path = "../Data/subsets/ben_1000_val_up_ran.jsonl"
geo_terms = load_geo_terms("results/geo_terms.txt")
access_token = os.getenv("HF_TOKEN")

n_loc = 100
precisions = [0, 1, 2, 3, 4, 5, 6, 7]
base_eval_dir = "results/experiment_1_R1" #### choose round 1 or 2

prompt_file_paths_R1 = ["prompts task #2/prompt1.txt",
                         "prompts task #2/prompt2.txt",
                         "prompts task #2/prompt3.txt",
                         "prompts task #2/prompt4.txt",
                         "prompts task #2/prompt5.txt",
                         "prompts task #2/prompt6.txt",
                         "prompts task #2/prompt7.txt",
                         "prompts task #2/prompt8.txt"]


prompt_file_paths_R2 = ["prompts task #2.0/prompt4.txt",
                         "prompts task #2.0/prompt5.txt",
                         "prompts task #2.0/prompt6.txt",
                         "prompts task #2.0/prompt7.txt",
                         "prompts task #2.0/prompt8.txt",
                         "prompts task #2.0/prompt9.txt",
                         "prompts task #2.0/prompt10.txt",
                         "prompts task #2.0/prompt11.txt"]

prompt_list_des = []
for p in prompt_file_paths_R1: #### choose round 1 or 2
    prompt_list_des.append(read_prompt(p))

num_prompts = len(prompt_file_paths_R1) 
# print(prompt_list_des)

# === GENERATION ===
for precision in precisions:
    print(f"\n=== Running for precision {precision} ===", flush=True)

    # Create subdirectory per precision
    eval_dir = os.path.join(base_eval_dir, f"precision_{precision}")

    # saves response, geo term count and true labels
    verbosity_test_up(
        prompts=prompt_list_des,
        geo_terms=geo_terms,
        data_path=data_path,
        n_loc=n_loc,
        model_name=model_name,
        token=access_token,
        eval_dir=eval_dir,
        precision=precision
    )

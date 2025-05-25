import sys
import json
import re
import torch
from openai import OpenAI, APIError, AuthenticationError
from math import radians, sin, cos, sqrt, atan2
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torchgeo.datasets import BigEarthNet
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pyproj import Proj, Transformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, SimilarityFunction
import nltk
import jsonlines
from collections import Counter
import textwrap
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_score, recall_score, f1_score


def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    places = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                places.append(json.loads(line.strip()))  # Now with .strip()!
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line.strip()}")
    return places


def read_prompt(file_path):
    """
    Reads a text file and returns its content as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


# def insert_coordinates(prompt: str, place: dict, placeholder: str = "{coords}"):
#   """ 
#   Inserts latitude and longitude into the given prompt at a specific placeholder.
#   """
#   lat = place['latitude']
#   lon = place['longitude']
#   coords_str = f"{lat}, {lon}"
#   return prompt.replace(placeholder, coords_str)


def insert_coordinates(prompt: str, place: dict, precision, key: str = "center", placeholder: str = "{coords}"):
    """ 
    Inserts latitude and longitude into the given prompt at a specific placeholder.
    ! updated to work with data consisting of multiple coordinate types (center, upper_left and lower_right) 
    """
    lat = place[key].get("latitude")
    lon = place[key].get("longitude")
    
    coords_str = f"{round(lat, precision)}, {round(lon, precision)}"
    
    return prompt.replace(placeholder, coords_str)
    


def insert_place(prompt: str, place: dict, placeholder: str = "{place}"):
  """ 
  Inserts the name of a place into the given prompt at a specific placeholder.
  """
  return prompt.replace(placeholder, place["name"])


def generate_response_deepseek(prompt: str):
  ak = "sk-c081fe785d5a4acd8fd4e7a73d227222"
  client = OpenAI(api_key=ak, base_url="https://api.deepseek.com")

  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=[
          {"role": "system", "content": "You are a helpful assistant"},
          {"role": "user", "content": prompt},
      ],
      stream=False,
      temperature=0.5,
      max_tokens=150
  )
  return response.choices[0].message.content

# old version
def load_model(name: str, token: str):
    """
    load model and tokenizer using huggingface
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, token=token)

    tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer

# updated version 
def load_model_up(name: str, token: str):
    """
    Load Meta's LLaMA 3 Instruct model + tokenizer from Hugging Face 
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, token=token, use_fast=True)

    # Ensure pad_token is set (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model 
    model = AutoModelForCausalLM.from_pretrained(
        name,
        token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # automatically spreads across available GPUs
    )

    return model, tokenizer



# def generate_response(model, tokenizer, prompt_w_coor: str): # response_file_path
#     """
#     Chat with the model.
#     """
#     messages = [
#       {"role": "system", "content": "You are a helpful AI assistant, expert in geography. Please return your answer in 100 tokens or less"},
#       {"role": "user", "content": prompt_w_coor}
#     ]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(**model_inputs, max_new_tokens=200, temperature = 0.0001)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     # # Create a dictionary to store the result
#     # result = {
#     #     "prompt": prompt_w_coor,
#     #     "response": response
#     # }

#     # # Append the response to the JSONL file
#     # with open(response_file_path, "a") as f:
#     #     f.write(json.dumps(result) + "\n")
    
#     return response

def generate_responses(model, tokenizer, prompts: list, max_tokens=200):
    """
    Batch chat with the model. Each prompt in `prompts` should be a string.
    Returns a list of responses, one per prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create list of message templates
    messages_batch = [
        [
            {"role": "system", "content": "You are a helpful AI assistant, expert in geography. Please return your answer in 100 tokens or less"},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    # Tokenize and prepare inputs
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_batch]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate responses in batch
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=0.0001
    )
    # Slice out only the generated part (after input prompt)
    generated_ids = [
        output_ids[input_ids.shape[-1]:]  # slice out the original prompt
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode only the new tokens
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return responses


# new version (detect leaks)
def generate_responses_up(model, tokenizer, prompts: list, max_tokens=200):
    """
    Batch chat with LLaMA 3 Instruct model. Each prompt in `prompts` should be a string.
    Returns a list of generated responses, one per prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare chat messages with system + user prompt
    messages_batch = [
        [
            {"role": "system", "content": "You are a helpful AI assistant, expert in geography. Please return your answer in 100 tokens or less."},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    # Apply chat template to get proper model input strings
    input_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    # Tokenize with padding and truncation
    encoded_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # Save prompt lengths to slice outputs later
    prompt_lengths = (attention_mask > 0).sum(dim=1).tolist()

    # Warn if anything was truncated
    if any(len(ids) >= tokenizer.model_max_length for ids in input_ids):
        print("Warning: Some inputs may have been truncated.")

    # Generate outputs
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.0001,
            do_sample=True  # deterministic output for testing
        )

    # Slice out only new tokens after the prompt
    new_tokens = [
        output[prompt_len:] for output, prompt_len in zip(generated_ids, prompt_lengths)
    ]

    # Decode just the new content
    responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return responses



def extract_lat_lon(text: str):
    """
    Extracts latitude from response. Please change the regex so it fits the llm - they respond differently!
    If latitude is South (S), it will be negative.
    If longitude is West (W), it will be negative.
    """
    lat_pattern = r"([-+]?\d+\.\d+)°?\s*([NnSs])" # Latitude:\s*([-+]?\d+\.\d+)°?\s*[NnSs]?
    lon_pattern = r"([-+]?\d+\.\d+)°?\s*([EeWw])" # Longitude:\s*([-+]?\d+\.\d+)°?\s*[EeWw]?

    match_lat = re.search(lat_pattern, text)
    match_lon = re.search(lon_pattern, text)

    lat = float(match_lat.group(1)) if match_lat else None
    lon = float(match_lon.group(1)) if match_lon else None

    # Apply negative sign if South or West
    if match_lat and match_lat.group(2).upper() == "S":
        lat = -lat
    if match_lon and match_lon.group(2).upper() == "W":
        lon = -lon

    return lat, lon


def add_and_save_data(places_data, file_path):
    """
    Adds the updated data to the JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for place in places_data:
            file.write(json.dumps(place) + "\n")
    print("Data saved successfully!")


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a)) 

    distance = 6371 * c  # Radius of Earth in km
    return distance


def load_geo_terms(file_path):
    """ 
    Loads geographic terminology from txt file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file.readlines() if line.strip()]


def count_geo_terms(description, geo_terms):
    """
    Counts which terms are being used in a describtion and how many. 
    """
    description = description.lower()  # Convert to lowercase
    used_terms = [term for term in geo_terms if term in description]
    count = len(used_terms)
    
    return count, used_terms 


# ---------------------------- dataset (BigEarthNet) -----------------------------------

class BigEarthNetS1Custom(Dataset):
    def __init__(self, root):
        """
        Args:
            root (str): Root directory containing the dataset.
        """
        self.root = root
        # self.patch_folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))] # includes hidden folders, so does not work?
        self.patch_folders = [
            f for f in os.listdir(root)
            if os.path.isdir(os.path.join(root, f)) and not f.startswith('.') and '.ipynb_checkpoints' not in f
        ]

        self.transform = transforms.ToTensor()  # Apply transformation (e.g., ToTensor)

    def __len__(self):
        return len(self.patch_folders)

    def __getitem__(self, idx):
        """
        Returns:
            image: A tensor representation of the VV and VH images combined.
            metadata: A dictionary with the extracted metadata.
        """
        # Get the folder for the current patch
        patch_folder = os.path.join(self.root, self.patch_folders[idx])
        
        # Define paths to images and metadata
        vv_image_path = os.path.join(patch_folder, f'{self.patch_folders[idx]}_VV.tif')
        vh_image_path = os.path.join(patch_folder, f'{self.patch_folders[idx]}_VH.tif')
        metadata_path = os.path.join(patch_folder, f'{self.patch_folders[idx]}_labels_metadata.json')

        print(metadata_path)

        # Load VV and VH images
        vv_image = Image.open(vv_image_path)
        vh_image = Image.open(vh_image_path)
        
        # Combine both images into one tenso
        vv_tensor = self.transform(vv_image)
        vh_tensor = self.transform(vh_image)

        # Stack VV and VH to create a multi-channel tensor
        image = torch.stack([vv_tensor, vh_tensor], dim=0)

        # Load metadata (labels, coordinates, etc.)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract relevant metadata
        labels = metadata.get('labels', [])
        coordinates = metadata.get('coordinates', {})
        projection = metadata.get('projection', '')
        corresponding_s2_patch = metadata.get('corresponding_s2_patch', '')
        scene_source = metadata.get('scene_source', '')
        acquisition_time = metadata.get('acquisition_time', '')

        # Package metadata in a dictionary
        metadata_dict = {
            'labels': labels,
            'coordinates': coordinates,
            'projection': projection,
            'corresponding_s2_patch': corresponding_s2_patch,
            'scene_source': scene_source,
            'acquisition_time': acquisition_time
        }

        return image, metadata_dict


def extract_utm_zone(projection_str):
    """
    Extract UTM zone from the PROJCS string in the projection metadata.
    
    Args:
        projection_str (str): The projection string from metadata.
        
    Returns:
        int: The UTM zone number.
    """
    # Use regular expression to extract the EPSG code
    match = re.search(r'UTM zone (\d+)', projection_str)
    
    if match:
        # Extract the UTM zone number
        utm_zone = int(match.group(1))  # group(1) is the captured UTM zone number
        return utm_zone
    return None  # Return None if no UTM zone is found


def utm_to_decimal(ulx, uly, lrx, lly, zone):
    """
    Convert UTM bounding box corners to decimal degrees (WGS84) using pyproj.Transformer.
    
    Args:
        ulx (float): Upper-left X coordinate (easting)
        uly (float): Upper-left Y coordinate (northing)
        lrx (float): Lower-right X coordinate (easting)
        lly (float): Lower-right Y coordinate (northing)
        zone (int): UTM zone number (e.g., 33 for UTM zone 33N)
        
    Returns:
        tuple: (center_deg_y, center_deg_x) in decimal degrees
    """
    
    # # Define the UTM projection for the specified zone
    # utm_proj = Proj(proj='utm', zone=zone, ellps='WGS84')
    
    # # Define the WGS84 (latitude, longitude) projection
    # wgs84_proj = Proj(proj='latlong', datum='WGS84')

    # # Create a transformer to convert from UTM to WGS84
    # transformer = Transformer.from_proj(utm_proj, wgs84_proj)
    # Convert UTM to WGS84 (decimal degrees)
    # ulx_deg, uly_deg = transformer.transform(ulx, uly)
    # lrx_deg, lly_deg = transformer.transform(lrx, lly)

    # Define transformer from UTM to WGS84 (lat/lon)
    epsg_code = f"326{zone:02d}"  # Assuming northern hemisphere
    print(epsg_code)
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

    # Transform corners
    ul_lon, ul_lat = transformer.transform(ulx, uly)
    lr_lon, lr_lat = transformer.transform(lrx, lly)

    # Transform center
    center_x = (ulx + lrx) / 2
    center_y = (uly + lly) / 2
    center_lon, center_lat = transformer.transform(center_x, center_y)

    return (ul_lon, ul_lat, lr_lon, lr_lat, center_lon, center_lat)



def save_samples_to_jsonl(dataset, num_samples, output_file):
    """
    Extracts a specified number of samples from the dataset, converts UTM coordinates
    to decimal degrees, and saves them along with labels to a JSONL file.

    Args:
        dataset (Dataset): An instance of BigEarthNetS1Custom.
        num_samples (int): Number of samples to process.
        output_file (str): Path to the JSONL output file.
    """
    data_list = []

    # random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i in indices: # for i in range(min(num_samples, len(dataset))):  # Ensure we don't exceed dataset size
        image, metadata = dataset[i]

        # Extract UTM coordinates
        coordinates = metadata.get('coordinates', {})
        ulx, uly = coordinates.get('ulx', None), coordinates.get('uly', None)
        lrx, lly = coordinates.get('lrx', None), coordinates.get('lly', None)

        if None in (ulx, uly, lrx, lly):
            print(f"Skipping sample {i} due to missing coordinates")
            continue  # Skip if any coordinate is missing

        # Extract UTM zone from projection metadata
        projection_str = metadata.get('projection', '')
        utm_zone = extract_utm_zone(projection_str)

        if utm_zone is None:
            print(f"Skipping sample {i} due to missing UTM zone")
            continue  # Skip if no UTM zone is found

        # Convert UTM to decimal degrees
        ul_lon, ul_lat, lr_lon, lr_lat, center_lon, center_lat = utm_to_decimal(ulx, uly, lrx, lly, utm_zone)

        # Save relevant metadata
        sample_data = {
            "center": {
                "latitude": center_lat,
                "longitude": center_lon
            },
            "upper_left": {
                "latitude": ul_lat,
                "longitude": ul_lon
            },
            "lower_right": {
                "latitude": lr_lat,
                "longitude": lr_lon
            },
            "labels": metadata.get('labels', [])
        }

        data_list.append(sample_data)

    # Write data to JSONL file
    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + "\n")

    print("All saved")


# ----------------------- STATISTICS -----------------------------
from pathlib import Path
from collections import defaultdict






def analyze_prompt_results(eval_dir, num_prompts, output_file=None):
    """
    Analyze the statistics of geoterm counts for different prompts and save results.

    Args:
        eval_dir (str): Directory containing the JSONL result files.
        num_prompts (int): Number of different prompts tested.
        output_file (str): Name of the output JSONL file.

    Returns:
        dict: Statistics for each prompt, sorted by mean count.
    """
    prompt_stats = []
    output_path = os.path.join(eval_dir, output_file)

    for p_idx in range(1, num_prompts + 1):
        file_path = os.path.join(eval_dir, f"results_prompt_{p_idx}.jsonl")

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue

        counts = []

        # Read the JSONL file
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                counts.append(data["count"])

        if counts:
            stats = {
                "prompt_id": int(p_idx), 
                "mean": float(np.mean(counts)),
                "median": float(np.median(counts)),
                "std": float(np.std(counts)),
                "max": int(np.max(counts)),
                "min": int(np.min(counts)),
                "total": int(np.sum(counts)),
                "num_samples": int(len(counts)),
            }
            prompt_stats.append(stats)


    # save statistics
    if output_file:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for i, stats in enumerate(prompt_stats):
                stats_str = json.dumps(stats, ensure_ascii=False)
                out_file.write(stats_str)
                
                # Add a newline after each entry 
                if i < len(prompt_stats) - 1:  # Ensure no extra newline after the last entry
                    out_file.write("\n")
    
        print(f"Statistics saved to {output_path}")

    # Sort prompts by mean count (descending order)
    prompt_stats = sorted(prompt_stats, key=lambda x: x["mean"], reverse=True)

    # Print top 2 prompts
    print("\n Top 2 Prompts Based on Mean Count:")
    for rank, stats in enumerate(prompt_stats[:2], start=1):
        print(f"Rank {rank}: Prompt {stats['prompt_id']} - Mean Count: {stats['mean']:.2f}")

    return prompt_stats


def plot_mean_counts(stats):
    sns.set(style="whitegrid")

    # Sort by prompt_id instead of mean
    stats_sorted = sorted(stats, key=lambda x: x["prompt_id"])

    prompts = [f"Prompt {s['prompt_id']}" for s in stats_sorted]
    means = [s["mean"] for s in stats_sorted]

    # Create a DataFrame for plotting
    df = pd.DataFrame({"Prompt": prompts, "Mean": means})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Prompt", y="Mean", hue="Prompt", palette="viridis", legend=False)

    plt.title("Mean Geoterm Count per Prompt")
    plt.ylabel("Mean Geoterm Count")
    plt.xlabel("Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("mean_counts_verbosity1.png")


def plot_boxplot(all_counts_dict):
    """
    Plots a boxplot of geoterm counts for each prompt.
    
    Args:
        all_counts_dict (dict): {prompt_id: [count1, count2, ...]}
    """
    # Flatten into a DataFrame
    data = []
    for pid, counts in all_counts_dict.items():
        for count in counts:
            data.append({"Prompt": f"Prompt {pid}", "Count": count})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="Prompt",
        y="Count",
        hue="Prompt",        # fix for future seaborn
        palette="coolwarm",
        legend=False         # prevent duplicate legend
    )

    plt.title("Distribution of Geoterm Counts by Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bloxplot_verbosity1.png")


def build_all_counts_dict(eval_dir, num_prompts):
    """
    Builds a dictionary of all geoterm counts per prompt from JSONL files.

    Returns:
        dict: {prompt_id: [count1, count2, ...], ...}
    """
    all_counts = {}

    for p_idx in range(1, num_prompts + 1):
        file_path = os.path.join(eval_dir, f"results_prompt_{p_idx}.jsonl")
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            continue

        counts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                counts.append(data.get("count", 0))  # default to 0 if missing

        all_counts[p_idx] = counts

    return all_counts



# ----------------------- EMBEDDINGS -----------------------------
def process_and_label_sentences(input_path, label_embeddings_path, output_path, model_name = "all-MiniLM-L6-v2", per_sentence = True):
    """
    Processes responses into embeddings, matches them to labels,
    and writes data to a .jsonl file.

    Args:
        input_path (str): Path to input JSONL with "response" fields.
        label_embeddings_path (str): Path to label embeddings JSONL.
        output_path (str): Output JSONL file path.
        model_name (str): SentenceTransformer model to use.
        per_sentence (bool): If True, embed each sentence. Otherwise, embed the full response.
    """
    # Load model
    model = SentenceTransformer(model_name)

    # Load data
    data = read_jsonl(input_path)
    label_entries = read_jsonl(label_embeddings_path)

    label_emb = np.array([np.array(e['embedding'], dtype=np.float32) for e in label_entries])
    labels = [e['label'] for e in label_entries]

    for item in data:
        response_text = item.get("response", "")

        if per_sentence:
            units = nltk.sent_tokenize(response_text)
        else:
            units = [response_text]
            
        unit_embeddings = model.encode(units)

        enriched = []
        
        for text, emb in zip(units, unit_embeddings):
            emb = np.array(emb, dtype=np.float32)

            similarities = [model.similarity(emb, label_vec) for label_vec in label_emb]
            
            indexed_similarities = list(enumerate(similarities))
            sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)

            top_matches = [
                {
                    "label": labels[idx],
                    "score": round(float(score), 4)
                }
                for idx, score in sorted_similarities[:3]
            ]
            
            text_key = "sentence" if per_sentence else "text"
            enriched.append({
                text_key: text,
                "embedding": emb.tolist(),
                "top_labels": top_matches
            })

        item["sentence_analysis"] = enriched

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    write_jsonl(output_path, data)
    print(f"Processed {len(data)} entries. Output saved to {output_path}")


## Old stuff, afraid to delete ...
# def add_sentence_embeddings_to_descriptions(input_path: str, model_name: str = "all-MiniLM-L6-v2", save_path=None):
#     """
#     Splits text into sentences and create embeddings for all of them

#     Returns:
#         dict: location, response, labels and embeddings
#     """
#     # Load model
#     model = SentenceTransformer(model_name)

#     # helper function
#     def write_jsonl(data, path):
#         with jsonlines.open(path, mode='w') as writer:
#             writer.write_all(data)

#     # Load original data
#     data = read_jsonl(input_path)

#     # Process each item
#     for item in data:
#         response_text = item.get("response", "")

#         # 1. Sentence splitting
#         sentences = nltk.sent_tokenize(response_text)

#         # 2. Embedding
#         embeddings = model.encode(sentences)

#         # 3. Add to entry
#         #item["embeddings"] = [emb.tolist() for emb in embeddings]
#         item["sentence_embeddings"] = [{"sentence": sent, "embedding": emb.tolist()} for sent, emb in zip(sentences, embeddings)
#     ]

#     # Save output
#     if save_path:
#         write_jsonl(data, save_path)
#         print(f" Finished processing {len(data)} entries.")
#         print(f" Output saved to: {save_path}")

#     return data



# def get_best_matching_labels_with_scores(input_path, embedding_l_path, output_path, model_name: str = "all-MiniLM-L6-v2"):
#     """
#     Finds the best matching label for each sentence

#     Returns:
#         best label and similarity score
#     """
#     # Load model
#     model = SentenceTransformer(model_name)
#     model.similarity_fn_name = SimilarityFunction.COSINE
#     print(model.similarity_fn_name)
    
#     sentence_embeddings = add_sentence_embeddings_to_descriptions(input_path)

#     write_jsonl(sentence_embeddings, output_path)
#     print(f" Finished processing {len(data)} entries.")
#     print(f" Output saved to: {save_path}")

#     # Load and prepare label embeddings
#     label_entries = read_jsonl(embedding_l_path)
#     label_emb = np.array(
#         [np.array(entry['embedding'], dtype=np.float32) for entry in label_entries]
#     )
#     labels = [entry['label'] for entry in label_entries]

#     predicted = []

#     for entry in sentence_embeddings:
#         #sent_embs = [np.array(emb, dtype=np.float32) for emb in entry['embeddings']]
#         sent_embs = [np.array(e['embedding'], dtype=np.float32) for e in entry['sentence_embeddings']]

#         location_predictions = []

#         for sent_emb in sent_embs:
#             similarities = [model.similarity(sent_emb, label_vec) for label_vec in label_emb]
#             # Combine similarities with their indices and sort by similarity score
#             indexed_similarities = list(enumerate(similarities))  # [(index, similarity_score), ...]
#             sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)  # Sort in descending order
            
#             # Get the top 3 entries
#             top_matches = [
#                 {
#                     "label": labels[idx],
#                     "score": round(float(score), 4)
#                 }
#                 for idx, score in sorted_similarities[:3]  # Take the top 3
#             ]
            
#             # Append to location_predictions
#             location_predictions.append(top_matches)

#         predicted.append(location_predictions)


#     original_data = read_jsonl(input_path)

#     for entry, preds in zip(original_data, predicted):
#         for sent_entry, pred_list in zip(entry["sentence_embeddings"], preds):
#             sent_entry["top_predictions"] = pred_list  # Store list of top 3 predictions

#     write_jsonl(output_path, original_data)
#     print(f"Output saved to: {output_path}")
#     #return all_best_labels



def write_jsonl(file_path, data):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            # Each dictionary gets written as a single line
            json.dump(item, f)
            f.write("\n")


def embed_labels():
    # Open the input labels.jsonl file and output file
    input_file = 'Data/labels.jsonl'
    output_file = 'Data/labels_with_embeddings.jsonl'
    
    # Read the original file and generate embeddings
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            label = obj.get("label", "")  # Adjust the key based on your JSON structure
            
            # Generate embedding for the label
            embedding = model.encode(label).tolist()  # Convert to list to store it in JSON
    
            # Add the embedding to the dictionary
            obj['embedding'] = embedding
    
            # Write the updated object with the embedding
            writer.write(obj)
    
    print(f"Embeddings saved in {output_file}")


# ----------------------- VISUALIZE RESULTS -----------------------------
def wrap_labels(labels, width=10):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]


# def evaluate_top3_predictions(jsonl_path, top_k=3, plot=True):
#     """
#     Evaluating the predicted labels by comparing the top three to the true labels.

#     Returns: 
#     Accuracy = #hits / #total pred
#     """
#     data = read_jsonl(jsonl_path)

#     total = 0
#     correct = 0
#     predicted_label_counter = Counter()
#     true_label_counter = Counter()

#     for entry in data:
#         true_labels = set(entry.get("labels", []))
#         true_label_counter.update(true_labels)
#         #print(true_labels)

#         # collect all predicted labels
#         all_predictions = []
        
#         for sent_entry in entry.get("sentence_embeddings", []):
#             for pred in sent_entry.get("top_predictions", []):
#                 all_predictions.append((pred["label"], pred["score"]))
#         #print(all_predictions) 
        
#         #  sort them by score
#         seen = set()
#         top_preds = []
#         for label, score in sorted(all_predictions, key=lambda x: x[1], reverse=True):
#             if label not in seen:
#                 top_preds.append(label)
#                 seen.add(label)
#             if len(top_preds) == top_k:
#                 break
#         #print(top_preds)

#         total += 1
#         if any(label in true_labels for label in top_preds):
#             correct += 1

#         # Label distribution update
#         predicted_label_counter.update(top_preds)

#     accuracy = correct / total if total > 0 else 0
#     print(f"\nEvaluated {total} descriptions")
#     print(f"Top-{top_k} Description-Level Accuracy: {accuracy:.4f}")

#     # Plotting
#     if plot:
#         # Plot the distribution of predicted labels
#         top_k = 20
#         common_labels = [label for label, _ in predicted_label_counter.most_common(top_k)]
#         pred_vals = [predicted_label_counter[label] for label in common_labels]
#         true_vals = [true_label_counter.get(label, 0) for label in common_labels]
    
#         wrapped_labels = wrap_labels(common_labels)
    
#         x = np.arange(len(common_labels))
#         width = 0.35
    
#         plt.figure(figsize=(18, 6))
#         plt.bar(x - width/2, pred_vals, width, label='Predicted')
#         plt.bar(x + width/2, true_vals, width, label='True')
#         plt.xticks(x, wrapped_labels, rotation=0, ha='center')
#         plt.title("Top 20 Labels: Predicted vs True")
#         plt.ylabel("Count")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
        
#     return accuracy, predicted_label_counter, true_label_counter

from sklearn.preprocessing import MultiLabelBinarizer


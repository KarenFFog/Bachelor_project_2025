import os
import json
import torch
import re
from torch.utils.data import Dataset
from PIL import Image
from pyproj import Transformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
import torchvision.transforms.functional as TF
import random



# first step: create data class
class BigEarthNetS2Custom(Dataset):
    def __init__(self, root, selected_bands=None, target_size=(120, 120), transform=None, folder_list=None):
        self.root = root
        all_folders = [
            f.lstrip("./") for f in os.listdir(root)
            if os.path.isdir(os.path.join(root, f)) and not f.startswith('.') and '.ipynb_checkpoints' not in f
        ]

        # Filter by provided list
        if folder_list is not None:
            self.patch_folders = [f for f in all_folders if f in folder_list]
            #cleaned_list = [name.lstrip("./") for name in folder_list]
            #self.patch_folders = [f for f in all_folders if f in cleaned_list]
        else:
            self.patch_folders = all_folders

        self.selected_bands = selected_bands or [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B8A', 'B11', 'B12'
        ]
        self.transform = transform or transforms.ToTensor()
        self.target_size = target_size

    def __len__(self):
        return len(self.patch_folders)

    def __getitem__(self, idx):
        patch_folder_name = self.patch_folders[idx]
        patch_folder = os.path.join(self.root, patch_folder_name)

        band_tensors = []
        
        try:
            for band in self.selected_bands:
                band_path = os.path.join(patch_folder, f'{patch_folder_name}_{band}.tif')
                band_image = Image.open(band_path)
                band_image_resized = band_image.resize(self.target_size, Image.BILINEAR)
                band_tensor = self.transform(band_image_resized).float()
                band_tensors.append(band_tensor)

            image = torch.cat(band_tensors, dim=0)

            metadata_path = os.path.join(patch_folder, f'{patch_folder_name}_labels_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata_dict = {
                'labels': metadata.get('labels', []),
                'coordinates': metadata.get('coordinates', {}),
                'projection': metadata.get('projection', ''),
                'corresponding_s1_patch': metadata.get('corresponding_s1_patch', ''),
                'scene_source': metadata.get('scene_source', ''),
                'acquisition_time': metadata.get('acquisition_time', ''),
                'location_name': patch_folder_name
            }

            return image, metadata_dict

        except Exception as e:
            print(f"[WARN] Skipping {patch_folder_name}: {e}", flush=True)
            
            # Try a different index (wrap around)
            next_idx = (idx + 1) % len(self)

            # Only recurse if it's not the same index (to avoid infinite loop)
            if next_idx != idx:
                return self.__getitem__(next_idx)
            else:
                raise RuntimeError("No valid samples found in dataset.")



# second step - convert to lat lon
def extract_utm_zone(projection_str):
    match = re.search(r'UTM zone (\d+)', projection_str)
    return int(match.group(1)) if match else None


def utm_to_decimal(ulx, uly, lrx, lly, zone):
    epsg_code = f"326{zone:02d}"  # Northern hemisphere
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

    # Corners
    ul_lon, ul_lat = transformer.transform(ulx, uly)
    lr_lon, lr_lat = transformer.transform(lrx, lly)

    # Center
    center_x = (ulx + lrx) / 2
    center_y = (uly + lly) / 2
    center_lon, center_lat = transformer.transform(center_x, center_y)

    return (ul_lon, ul_lat, lr_lon, lr_lat, center_lon, center_lat)


def save_samples_to_jsonl(dataset, output_file):
    data_list = []

    for i in range(len(dataset)):
        image, metadata = dataset[i]

        # Required fields
        coordinates = metadata.get('coordinates', {})
        ulx, uly = coordinates.get('ulx'), coordinates.get('uly')
        lrx, lry = coordinates.get('lrx'), coordinates.get('lry')
        projection_str = metadata.get('projection', '')
        utm_zone = extract_utm_zone(projection_str)
        location_name = metadata.get('location_name', f"loc_{i}")  # <- Add this in your dataset class if needed

        if None in (ulx, uly, lrx, lry, utm_zone):
            print(f"[WARN] Skipping {location_name}: missing coordinates or UTM zone")
            continue

        ul_lon, ul_lat, lr_lon, lr_lat, center_lon, center_lat = utm_to_decimal(ulx, uly, lrx, lry, utm_zone)

        sample_data = {
            "location_name": location_name,
            "center": {"latitude": center_lat, "longitude": center_lon},
            "upper_left": {"latitude": ul_lat, "longitude": ul_lon},
            "lower_right": {"latitude": lr_lat, "longitude": lr_lon},
            "labels": metadata.get('labels', [])
        }

        data_list.append(sample_data)

    with open(output_file, 'w') as f:
        for entry in data_list:
            f.write(json.dumps(entry) + "\n")

    print(f"[DONE] Saved {len(data_list)} entries to {output_file}")





# third step - generating
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



def insert_coordinates(prompt: str, place: dict, precision, key: str = "center", placeholder: str = "{coords}"):
    """ 
    Inserts latitude and longitude into the given prompt at a specific placeholder.
    ! updated to work with data consisting of multiple coordinate types (center, upper_left and lower_right) 
    """
    lat = place[key].get("latitude")
    lon = place[key].get("longitude")
    
    coords_str = f"{round(lat, precision)}, {round(lon, precision)}"
    
    return prompt.replace(placeholder, coords_str)



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



def generate_responses_up(model, tokenizer, prompts: list, max_tokens=200):
    """
    Batch chat with LLaMA 3 Instruct model. Each prompt in `prompts` should be a string.
    Returns a list of generated responses, one per prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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


def run_single_prompt_old(prompt_template, data_path, n_loc, model_name, token, output_file, precision, batch_size=32):
    model, tokenizer = load_model_up(model_name, token)
    data_points = read_jsonl(data_path)
    data_points = data_points[:min(n_loc, len(data_points))]

    results = []

    for i in range(0, len(data_points), batch_size):
        batch = data_points[i:i + batch_size]
        prompt_batch = [
            insert_coordinates(prompt_template, loc, precision)
            for loc in batch
        ]

        print(f"Generating batch {i}–{i + len(batch)}...")
        batch_responses = generate_responses_up(model, tokenizer, prompt_batch)

        for loc, resp in zip(batch, batch_responses):
            results.append({
                "location": loc["location_name"],
                "description": resp.strip()
            })

    # Save all at once
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[DONE] Saved {len(results)} descriptions to {output_file}")


def run_single_prompt(prompt_template, data_path, n_loc, model_name, token, output_file, precision, batch_size=32, resume=True):
    model, tokenizer = load_model_up(model_name, token)
    data_points = read_jsonl(data_path)
    data_points = data_points[:min(n_loc, len(data_points))]

    # If resuming, check how many lines already exist
    already_done = 0
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            already_done = sum(1 for _ in f)
        print(f"[INFO] Resuming from index {already_done}")

    # Open file in append mode
    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(already_done, len(data_points), batch_size):
            batch = data_points[i:i + batch_size]
            prompt_batch = [
                insert_coordinates(prompt_template, loc, precision)
                for loc in batch
            ]

            print(f"Generating batch {i}–{i + len(batch)}...")
            batch_responses = generate_responses_up(model, tokenizer, prompt_batch)

            for loc, resp in zip(batch, batch_responses):
                f.write(json.dumps({
                    "location": loc["location_name"],
                    "description": resp.strip()
                }) + "\n")

    print(f"[DONE] Finished writing to {output_file}")

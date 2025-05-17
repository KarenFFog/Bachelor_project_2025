import json
import re
from gen import *
from openai import OpenAI
from math import radians, sin, cos, sqrt, atan2


def generate_responses_for_all_places(prompt, f_file_path, t_file_path, r_file_path, model_name, token):
    """
    Generates responses for all places in the data,
    calculates the distance between the actual and predicted coordinates,
    and saves the updated data.
    """
    # load model
    model, tokenizer = load_model(model_name, token)

    # Move model to GPU explicitly - don't know if this is nessesary..
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # read data
    places_data = read_jsonl(f_file_path)

    # generate coordinates 
    for place in places_data:
        
        prompt_with_coords = insert_place(prompt, place)
        
        response = generate_response(model, tokenizer, prompt_with_coords, r_file_path)
        #print(response)
        
        pred_lat, pred_lon = extract_lat_lon(response)
        #print(pred_lat, pred_lon)
        
        place["predicted_latitude"] = pred_lat
        place["predicted_longitude"] = pred_lon

        # calculate distance
        lat1, lon1 = place.get("latitude"), place.get("longitude")
        lat2, lon2 = place.get("predicted_latitude"), place.get("predicted_longitude")

        # Ensure all coordinates are valid before calculating distance
        if None not in (lat1, lon1, lat2, lon2):
            place["distance"] = haversine(lat1, lon1, lat2, lon2)
        else:
            place["distance"] = None 
    
    # Save the updated data
    add_and_save_data(places_data, t_file_path)


def eval_task1(t_file_path):
    total = 0
    count = 0
    no_answer_count = 0 # how many times does it not answer the prompt
    with open(t_file_path, "r", encoding="utf-8") as file:
        try:
            for line in file:
                data = json.loads(line)  # Load each JSON object
                if "distance" in data and isinstance(data["distance"], (int, float)):  # Ensure it's a number
                    total += data["distance"]
                    count += 1
                else:
                    no_answer_count +=1
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")  # Debugging
    if count == 0:
        return None
    
    average_score = total/count
    return average_score, no_answer_count



def evaluate_multiple_prompts(prompts, f_file_path, t_file_paths, r_file_path, model_name, token, task_1_avg):
    """
    Runs the response generation and evaluation for multiple prompts and datasets.
    Saves all average scores in a single JSONL file.
    """
    results = []  # Store results for all prompts

    for i in range(len(prompts)):  
        print(f"Processing prompt {i+1}/{len(prompts)}...")

        # Step 1: Generate responses
        generate_responses_for_all_places(prompts[i], f_file_path, t_file_paths[i], r_file_path, model_name, token)

        # Step 2: Calculate the average distance
        avg_distance, no_answer_count = eval_task1(t_file_paths[i])

        # Step 3: Store results
        results.append({
            "prompt": prompts[i],
            "model": model_name,
            "t_file_path": t_file_paths[i],
            "average_distance": avg_distance,
            "no_answer_count": no_answer_count
        })

    # Step 4: Save results to JSONL
    with open(task_1_avg, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"All results saved to {task_1_avg}.")

    
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# TEST 2

def generate_description_for_all_coords(prompts, geo_file_path, f_file_path, t_file_path, r_file_path, model_name, token, coords, task_2_eval):
    # load model
    model, tokenizer = load_model(model_name, token)

    # Move model to GPU explicitly - don't know if this is nessesary..
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # read geo terms
    geo_terms = load_geo_terms(geo_file_path)

    # store results for different prompts
    results = []
    
    # generate descriptions
    for i in range(len(prompts)):
        #print(coords)
        prompt_with_coords = insert_coordinates(prompts[i], coords)
        print(f"prompt: {prompt_with_coords}")
        
        response = generate_response(model, tokenizer, prompt_with_coords, r_file_path)
        #print(response)

        count, matches = count_geo_terms(response, geo_terms)

        results.append({
            "prompt": prompts[i],
            "response": response,
            "model": model_name,
            "count": count,
            "matches": matches
        })
        
        print(f"Description: {response}")
        print(f"Geographical term count: {count}")
        print(f"Matched terms: {', '.join(matches) if matches else 'None'}\n")

    # Step 4: Save results to JSONL
    with open(task_2_eval, "w", encoding="utf-8") as file:
        for res in results:
            file.write(json.dumps(res) + "\n")
    
    print(f"All results saved to {task_2_eval}.")



def verbosity_test(prompt, geo_terms, data_path, n_loc, model_name, token, eval_path):
    """
    Prompt model with the same prompt for n different locations, save geoterm count
    
    prompt: 
    geo_terms: a list of geological terms
    data_path: path to data jsonl with coordinates
    model_name: llama 3.2
    token: hugging face token
    eval_path: to save responses and results in
    """
    
    # load model
    model, tokenizer = load_model(model_name, token)

    # Move model to GPU explicitly - don't know if this is nessesary..
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load data
    data_points = read_jsonl(data_path)

    # store results for different prompts
    results = []
    
    # generate descriptions
    for coords in range(n_loc):
        #print(coords)
        prompt_with_coords = insert_coordinates(prompt, data_points[coords])
        print(f"prompt: {prompt_with_coords}")
        
        response = generate_response(model, tokenizer, prompt_with_coords)
        #print(response)

        count, matches = count_geo_terms(response, geo_terms)

        results.append({
            "response": response,
            "count": count
        })

    # Step 4: Save results to JSONL
    with open(eval_path, "w", encoding="utf-8") as file: # change "w" to "a" if you want to append instead of overwrite
        for res in results:
            file.write(json.dumps(res) + "\n")
    
    print(f"All results saved to {eval_path}.")




# def verbosity_test_up(prompts, geo_terms, data_path, n_loc, model_name, token, eval_dir):
#     """
#     Prompt model with multiple prompts for n different locations, save results separately.

#     Args:
#         prompts (list): List of prompt templates (e.g., 8 different prompts).
#         geo_terms (list): List of geological terms.
#         data_path (str): Path to data JSONL file with coordinates.
#         n_loc (int): Number of locations to test.
#         model_name (str): Model name (e.g., "llama 3.2").
#         token (str): Hugging Face token.
#         eval_dir (str): Directory to save separate result files.
#     """

#     # Load model
#     model, tokenizer = load_model(model_name, token)

#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load data
#     data_points = read_jsonl(data_path)

#     # Ensure n_loc does not exceed available data
#     n_loc = min(n_loc, len(data_points))

#     for p_idx, prompt in enumerate(prompts):
#         results = []
#         output_file = f"{eval_dir}/results_prompt_{p_idx+1}.jsonl"

#         for i in range(n_loc):
#             # Insert coordinates into the prompt
#             prompt_with_coords = insert_coordinates(prompt, data_points[i])

#             print(f"Processing Prompt {p_idx+1}, Location {i+1}: {prompt_with_coords}")

#             # Generate response
#             response = generate_response(model, tokenizer, prompt_with_coords)

#             if response is not None:
#                 count, matches = count_geo_terms(response, geo_terms)
#             else:
#                 count, matches = 0, []

#             # Store results
#             results.append({
#                 "location": i+1,
#                 "response": response,
#                 "count": count
#             })

#         # Save results for this prompt in a separate JSONL file
#         with open(output_file, "w", encoding="utf-8") as file:
#             for res in results:
#                 file.write(json.dumps(res) + "\n")

#         print(f"Results for Prompt {p_idx+1} saved to {output_file}.")


def verbosity_test_up(prompts, geo_terms, data_path, n_loc, model_name, token, eval_dir, precision):
    """
    Prompt model with multiple prompts for n different locations, save results separately.
    """

    # Load model
    model, tokenizer = load_model(model_name, token)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    data_points = read_jsonl(data_path)

    # Limit to available data
    n_loc = min(n_loc, len(data_points))
    data_points = data_points[:n_loc]

    os.makedirs(eval_dir, exist_ok=True)

    for p_idx, prompt_template in enumerate(prompts):
        print(f"\n Processing Prompt Template {p_idx + 1}...")

        prompt_batch = []
        for i in range(n_loc):
            filled_prompt = insert_coordinates(prompt_template, data_points[i], precision)
            prompt_batch.append(filled_prompt)

        print(f"Generating {len(prompt_batch)} responses in batch...")
        responses = generate_responses_up(model, tokenizer, prompt_batch)

        results = []
        for i, response in enumerate(responses):
            if response:
                count, matches = count_geo_terms(response, geo_terms)
            else:
                count, matches = 0, []

            # Extract labels from the data_point
            labels = data_points[i].get("labels", [])
            
            results.append({
                "location": i + 1,
                "response": response,
                "count": count,
                "true labels": labels
            })

        output_file = f"{eval_dir}/results_prompt_{p_idx + 1}.jsonl"
        with open(output_file, "w", encoding="utf-8") as file:
            for res in results:
                file.write(json.dumps(res) + "\n")

        print(f"Results saved to {output_file}")



# def generate_describtions(prompts, data_path, n_loc, model_name, token, eval_dir, add):
#     """
#     Prompt model with multiple prompts for n different locations, save results separately.

#     Args:
#         prompts (list): List of prompt templates (e.g., 8 different prompts).
#         data_path (str): Path to data JSONL file with coordinates.
#         n_loc (int): Number of locations to test.
#         model_name (str): Model name (e.g., "llama 3.2").
#         token (str): Hugging Face token.
#         eval_dir (str): Directory to save separate result files.
#     """

#     # Load model
#     model, tokenizer = load_model(model_name, token)

#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load data
#     data_points = read_jsonl(data_path)

#     # Ensure n_loc does not exceed available data
#     n_loc = min(n_loc, len(data_points))

#     for p_idx, prompt in enumerate(prompts):
#         results = []
#         output_file = f"{eval_dir}/results_best_prompt_2dec_{p_idx}_{add}.jsonl"

#         for i in range(n_loc):
#             # Insert coordinates into the prompt
#             prompt_with_coords = insert_coordinates(prompt, data_points[i])
#             labels = data_points[i]['labels']

#             print(f"Processing Prompt {p_idx+1}, Location {i+1}: {prompt_with_coords}")

#             # Generate response
#             response = generate_response(model, tokenizer, prompt_with_coords)

#             # Store results
#             results.append({
#                 "location": i+1,
#                 "response": response,
#                 "labels": labels,
#             })

#         # Save results for this prompt in a separate JSONL file
#         with open(output_file, "w", encoding="utf-8") as file:
#             for res in results:
#                 file.write(json.dumps(res) + "\n")

#         print(f"Results for Prompt {p_idx+1} saved to {output_file}.")


# EDITED - ORIGINAL ABOVE
def generate_descriptions(prompts, data_path, n_loc, model_name, token, eval_dir, add):
    """
    Prompt model with multiple prompts for n different locations, save results separately.
    """

    # Load model
    model, tokenizer = load_model(model_name, token)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    data_points = read_jsonl(data_path)

    # Ensure n_loc does not exceed available data
    n_loc = min(n_loc, len(data_points))
    data_points = data_points[:n_loc]

    for p_idx, prompt_template in enumerate(prompts):
        prompt_list = []
        labels_list = []

        for i in range(n_loc):
            # Prepare batch of prompts with coordinates
            filled_prompt = insert_coordinates(prompt_template, data_points[i])
            prompt_list.append(filled_prompt)
            labels_list.append(data_points[i]['labels'])

        print(f"Generating responses for Prompt {p_idx + 1} with {n_loc} locations...")
        # Generate responses in batch
        responses = generate_responses(model, tokenizer, prompt_list)

        # Prepare output
        results = []
        for i, (resp, labels) in enumerate(zip(responses, labels_list)):
            results.append({
                "location": i + 1,
                "response": resp,
                "labels": labels,
            })

        # Save to file
        output_file = f"{eval_dir}/results_best_prompt_2dec_{p_idx}_{add}.jsonl"
        os.makedirs(eval_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            for res in results:
                file.write(json.dumps(res) + "\n")

        print(f"Results for Prompt {p_idx + 1} saved to {output_file}")










def find_nearest_habitation(prompt_file_path, f_file_path, t_file_path):
    """
    Finds the nearest habitation for all places in the data,
    
    """
    # read data
    places_data = read_jsonl(f_file_path)
    prompt = read_prompt(prompt_file_path)

    for place in places_data:
        # generate coordinates 
        prompt_with_coords = insert_place(prompt, place)
        response = generate_response(prompt_with_coords)
        
        # extract the name of the nearest habitation
        # ??
    
    # Save the updated data
    add_and_save_data(places_data, t_file_path)


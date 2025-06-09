import rasterio
from rasterio.plot import show
from rasterio.warp import transform_bounds
import folium
from folium.raster_layers import ImageOverlay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import jsonlines
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path
from collections import defaultdict
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
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings


from eval import *
from gen import * 



def create_map_with_overlay(tif_path, center_lat, center_lon, save_img_path):
    """
    Creates an interactive folium map with a raster overlay and a marker at a specified center.

    Args:
        tif_path (str): Path to the GeoTIFF file.
        center_lat (float): Latitude of the center marker.
        center_lon (float): Longitude of the center marker.
        save_img_path (str): Path to save the resulting HTML map.
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        print("Bounds:", bounds)
        print("CRS:", crs)
    
        # Transform bounds to lat/lon
        bounds_latlon = transform_bounds(src.crs, "EPSG:4326", *bounds)
        print("Lat/Lon Bounds:", bounds_latlon)

    # Read data as image
    with rasterio.open(tif_path) as src:
        image = src.read(1)  # Read first band
        image = np.clip(image, np.percentile(image, 2), np.percentile(image, 98))  # stretch contrast
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype('uint8')
    
    # Convert image to PNG using matplotlib
    plt.imsave("temp.png", image, cmap="gray")
    
    # Use lat/lon bounds
    south, west, north, east = bounds_latlon[1], bounds_latlon[0], bounds_latlon[3], bounds_latlon[2]
    
    # Create map
    m = folium.Map(location=[(south+north)/2, (west+east)/2], zoom_start=10)
    
    folium.Marker(
        location=[center_lat, center_lon],
        popup="Center Point",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)
    
    # Add the overlay
    image_overlay = folium.raster_layers.ImageOverlay(
        name='VV Band',
        image='temp.png',
        bounds=[[south, west], [north, east]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )
    image_overlay.add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(save_img_path)
    print("saved image")



def visualize_label_embeddings(label_file_path):
    """
    Plots embedded labels from a JSONL file.

    Args:
        label_file_path (str): Path to the JSONL file containing data entries with a 'labels' and a 'embedding' field.     
    """
    labels = []
    embeddings = []
    
    with jsonlines.open(label_file_path) as reader:
        for entry in reader:
            labels.append(entry["label"])
            embeddings.append(np.array(entry["embedding"], dtype=np.float32))
    
    # Reduce dimensions for plotting using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    
    # Plotting
    plt.figure(figsize=(16, 12))
    for i, label in enumerate(labels):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.text(x + 0.5, y, label, fontsize=8)
    
    plt.title("t-SNE Visualization of Label Embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #plt.savefig("Figures/visualization of label embeddings.png")



def plot_true_label_distribution(jsonl_path, save_img_path, title="True Label Distribution", wrap_width=10):
    """
    Plots the distribution of true labels from a JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file containing data entries with a 'labels' field.
        save_img_path (str): Path to where you want the plot saved.
        title (str): Title of the plot.
        wrap_width (int): Maximum width for label wrapping.
    """
    data = read_jsonl(jsonl_path)

    # Count labels
    label_counter = Counter()
    for entry in data:
        labels = entry.get("labels", [])
        label_counter.update(labels)

    # # Print label counts
    # print("\nLabel Counts:")
    # for label, count in label_counter.items():
    #     print(f"{label}: {count}")

    labels = list(label_counter.keys())
    counts = list(label_counter.values())
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=wrap_width)) for label in labels]

    # Plot
    plt.figure(figsize=(26, 6))
    plt.bar(wrapped_labels, counts, color='skyblue')
    #plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    #plt.savefig(save_img_path)



def load_counts_per_group(eval_dir, precisions, num_prompts):
    """
    Loads geo term count values from result files across different prompts and precision levels.

    Args:
        eval_dir (str): Path to the base evaluation directory containing subfolders named by precision (e.g., 'precision_1').
        precisions (list): List of precision values to process (e.g., [1, 2, 3]).
        num_prompts (int): Number of prompts to iterate over.
    """

    eval_dir = Path(eval_dir)
    boxplot_data = {}

    for prompt_id in range(1, num_prompts + 1):
        for precision in precisions:
            label = f"Prompt {prompt_id}"
            result_file = eval_dir / f"precision_{precision}" / f"results_prompt_{prompt_id}.jsonl"
            if not result_file.exists():
                print("not found")
                continue

            with result_file.open("r", encoding="utf-8") as f:
                counts = [json.loads(line).get("geo term count", 0) for line in f]

            boxplot_data[label] = counts

    return boxplot_data




def plot_boxplot(count_data, labels, title, box_colors=None):
    """
    Plots boxplot of geographic term counts for different prompt variants.

    Args:
        count_data (dict): Dictionary where keys are prompt labels and values are lists of geographic term counts (could be made using load_counts_per_group)
        labels (list): List of labels for the x-axis corresponding to each box.
        title (str): Title of the plot.
        box_colors (list, optional): List of colors for the boxes.
    """
    
    data = list(count_data.values())
    num_boxes = len(data)

    plt.figure(figsize=(max(12, num_boxes * 1.2), 6))
    boxprops = dict(linewidth=1.5)

    # Create the boxplot
    bp = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=True, boxprops=boxprops)

    # Default to colorblind-friendly colors if none provided
    if box_colors is None:
        default_colors = plt.get_cmap("tab20").colors  # 20 distinct colors
        box_colors = [default_colors[i % len(default_colors)] for i in range(num_boxes)]

    # Apply colors to each box
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Optional: color medians or whiskers too
    for median in bp["medians"]:
        median.set_color("black")

    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Number of Geographic Terms", fontsize=16)
    plt.xlabel("Prompt Variant", fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def do_stats(eval_dir, precisions, num_prompts, output_file=None):
    """
    Compute and collect statistics for each prompt across precision levels.

    Args:
        eval_dir (str): Path to the base evaluation directory containing subfolders named by precision (e.g., 'precision_1').
        precisions (list): List of precision values to include in the analysis.
        num_prompts (int): Number of prompts to process
        output_file (str, optional): If provided, saves the computed statistics as a JSON file at this location.
    """
    
    eval_dir = Path(eval_dir)
    prompt_stats = []

    for prompt_id in range(1, num_prompts + 1):
        for precision in precisions:
            result_file = eval_dir / f"precision_{precision}" / f"results_prompt_{prompt_id}.jsonl"

            if not result_file.exists():
                print(f"Warning: File not found: {result_file}")
                continue

            counts = []
            with result_file.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    counts.append(data.get("geo term count", 0)) 

            if counts:
                stats = {
                    "prompt_id": prompt_id,
                    "precision": precision,
                    "mean": float(np.mean(counts)),
                    "median": float(np.median(counts)),
                    "std": float(np.std(counts)),
                    "max": int(np.max(counts)),
                    "min": int(np.min(counts)),
                    "total": int(np.sum(counts)),
                    "num_samples": len(counts),
                }
                prompt_stats.append(stats)

    # Optionally save stats to file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(prompt_stats, out, indent=2)

    return prompt_stats



def plot_mean_vs_precision(
    base_eval_dir, 
    prompt_names, 
    save_stats_path=None, 
    precisions=[0, 1, 2, 3, 4, 5, 6, 7], 
    title="Mean Count vs. Precision", 
    save_path=None,
    prompt_ids_to_plot=None  # New parameter
):
    """
    Plot mean term count as a function of precision for selected prompts.

    Args:
        base_eval_dir (str): Directory with evaluation results.
        prompt_names (list): Full list of prompt names (e.g., ['R1-P1', 'R2-P5', ...])
        save_stats_path (str): Optional path to save intermediate stats.
        precisions (list): List of precision values to include in the analysis.
        title (str): Title of the plot.
        save_path (str): Optional path to save the figure.
        prompt_ids_to_plot (list): List of prompt indices to include, e.g., [1, 5, 9]
    """

    stats = do_stats(base_eval_dir, precisions, len(prompt_names), save_stats_path)
    
    prompt_data = defaultdict(dict)
    for entry in stats:
        prompt_id = entry["prompt_id"]
        precision = entry["precision"]
        mean = entry["mean"]
        prompt_data[prompt_id][precision] = mean

    plt.figure(figsize=(10, 6))
    for prompt_id, values in sorted(prompt_data.items()):
        if prompt_ids_to_plot and prompt_id not in prompt_ids_to_plot:
            continue  # Skip prompts not selected

        sorted_precisions = sorted(values.keys())
        means = [values[p] for p in sorted_precisions]
        label = f"{prompt_names[prompt_id - 1]}"  # -1 since list is 0-based
        plt.plot(sorted_precisions, means, marker='o', label=label)

    plt.xlabel("Coordinate Precision")
    plt.ylabel("Mean Count of Geographic Terms")
    plt.title(title)
    plt.xticks(sorted({entry['precision'] for entry in stats}))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()



def plot_acc_vs_prec_one_prompt(acc_dir, precisions, ks, prompt):
    """
    Plots accuracy vs. precision level for different top-k predictions

    Args:
        precisions (list): List of precisions e.g. [0, 1, 2, 3, 4, 5]
        ks (list): List of top-k's
        prompt (int): number of prompt
    """
    
    acc_dir = Path(acc_dir)
    accuracy_matrix = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
    
        for precision in precisions:
            row = []
            for k in ks:
                path = acc_dir / f"precision_{precision}" / f"predicted_labels_{prompt}.jsonl"
                res = evaluate_top3_predictions(path, top_k=k, plot=False)
                row.append(res["accuracy"])
            accuracy_matrix.append(row)
        
        accuracy_matrix = np.array(accuracy_matrix)
    
    #print(accuracy_matrix)
    
    plt.figure(figsize=(8, 5))
    plt.style.use("seaborn-v0_8-colorblind")
    
    for i, k in enumerate(ks):
        plt.plot(precisions, accuracy_matrix[:, i], marker="o", label=f"Top-{k}")
    
    plt.xlabel("Coordinate Precision Level")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Precision Level for Different Top-K Predictions (using prompt 1)")
    plt.legend(title="Top-K")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_top3_predictions(jsonl_path, top_k=3, plot=True):
    """
    Evaluates top-k predicted labels against true labels for multi-label classification tasks.

    Args:
        jsonl_path (str): Path to a JSONL file where each line is a dictionary containing 'true labels' and 'sentence_analysis' with 'top_labels' predictions.
        top_k (int): Number of top predictions to consider for evaluation (default is 3).
        plot (bool): If True, displays a bar chart comparing the top predicted vs true labels.
    """
    
    data = read_jsonl(jsonl_path)

    total = 0
    correct = 0
    predicted_label_counter = Counter()
    true_label_counter = Counter()

    all_true_sets = []
    all_pred_sets = []

    for entry in data:
        true_labels = set(entry.get("true labels", []))
        #print(true_labels)
        true_label_counter.update(true_labels)
        all_true_sets.append(true_labels)
        
        all_predictions = []
        for sent_entry in entry.get("sentence_analysis", []): #("sentence_embeddings", []):
            #print(sent_entry)
            for pred in sent_entry.get("top_labels", []):
                all_predictions.append((pred["label"], pred["score"]))

        seen = set()
        top_preds = []
        for label, score in sorted(all_predictions, key=lambda x: x[1], reverse=True):
            if label not in seen:
                top_preds.append(label)
                seen.add(label)
            if len(top_preds) == top_k:
                break

        predicted_label_counter.update(top_preds)
        all_pred_sets.append(set(top_preds))

        total += 1
        if any(label in true_labels for label in top_preds):
            correct += 1

    accuracy = correct / total if total > 0 else 0

    # Binarize multilabel data
    mlb = MultiLabelBinarizer()
    y_true_binary = mlb.fit_transform(all_true_sets)
    y_pred_binary = mlb.transform(all_pred_sets)

    # Compute metrics
    precision = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)

    # print(f"\nEvaluated {total} descriptions")
    # print(f"Top-{top_k} Description-Level Accuracy: {accuracy:.4f}")
    # print(f"Micro Precision: {precision:.4f}")
    # print(f"Micro Recall:    {recall:.4f}")
    # print(f"Micro F1 Score:  {f1:.4f}")

    if plot:
        top_n = 20
        common_labels = [label for label, _ in predicted_label_counter.most_common(top_n)]
        pred_vals = [predicted_label_counter[label] for label in common_labels]
        true_vals = [true_label_counter.get(label, 0) for label in common_labels]

        wrapped_labels = wrap_labels(common_labels)

        x = np.arange(len(common_labels))
        width = 0.35

        plt.figure(figsize=(18, 6))
        plt.bar(x - width/2, pred_vals, width, label='Predicted')
        plt.bar(x + width/2, true_vals, width, label='True')
        plt.xticks(x, wrapped_labels, rotation=0, ha='center')
        plt.title("Top 20 Labels: Predicted vs True")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_label_counter": predicted_label_counter,
        "true_label_counter": true_label_counter
    }



def plot_acc_vs_prec_one_k(acc_dir, precisions, k, prompt_names, prompts_id):
    """
    Plots top-k accuracy versus precision level for different prompts.

    Args:
        acc_dir (str): Path to the base directory containing prediction results in subfolders named by precision level (e.g., 'precision_3').
        precisions (list): List of precision levels to evaluate (e.g., [0, 1, 2, 3, 4, 5]).
        k (int): Number of top predictions to consider when computing accuracy (e.g., top-3).
        prompt_names (list): List of prompt names.
        prompts_id (list): List of prompt IDs to include in the plot (e.g. [1, 3, 7]).
    """
    
    acc_dir = Path(acc_dir)
    accuracy_matrix = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        for prompt_id in prompts_id:
            row = []
            for precision in precisions:
                path = acc_dir / f"precision_{precision}" / f"predicted_labels_{prompt_id}.jsonl"
                res = evaluate_top3_predictions(path, top_k=k, plot=False)
                row.append(res["accuracy"])
            accuracy_matrix.append(row)
    
    accuracy_matrix = np.array(accuracy_matrix)  # shape: [n_prompts, n_precisions]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.style.use("seaborn-v0_8-colorblind")
    
    
    for i, prompt_id in enumerate(prompts_id):
        label = f"Prompt {prompt_names[prompt_id-1]}"
        plt.plot(precisions, accuracy_matrix[i], marker="o", label=label)
    
    plt.xlabel("Coordinate Precision Level")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Precision Level (Top-{k} Predictions)")
    plt.legend(title="Prompt")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def collect_accuracy_verbosity_data(base_dir_mean, base_dir_acc, prompt_ids, precisions, top_k=3, stats_path="stats_test.jsonl"):
    """
    Collects accuracy and verbosity statistics across multiple prompts and precision levels.

    Args:
        base_dir_mean (str): Path to the directory containing evaluation results used for computing verbosity stats.
        base_dir_acc (str): Path to the directory containing predicted label files for computing accuracy.
        prompt_ids (list): List of prompt IDs to include in the analysis.
        precisions (list): List of precision levels to evaluate (e.g., [0, 1, 2, 3]).
        top_k (int): Number of top predictions to consider when computing accuracy (default is 3).
        stats_path (str): Path to save the verbosity statistics as a JSON file.

    Returns:
        tuple: Four NumPy arrays of shape [n_prompts, n_precisions]:
            - accuracy_matrix (np.ndarray): Top-k accuracy values for each prompt/precision pair.
            - verbosity_matrix (np.ndarray): Mean "geo term count" (verbosity) for each prompt/precision pair.
            - accuracy_se_matrix (np.ndarray): Standard error of accuracy for each prompt/precision pair.
            - verbosity_std_matrix (np.ndarray): Standard deviation of "geo term count" for each prompt/precision pair.
    """
    
    verbosity_data = do_stats(base_dir_mean, precisions, len(prompt_ids), stats_path)

    acc_dir = Path(base_dir_acc)
    accuracy_matrix = []
    verbosity_matrix = []
    accuracy_se_matrix = []
    verbosity_std_matrix = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for prompt_id in prompt_ids:
            accuracy_row = []
            accuracy_se_row = []
            verbosity_row = []
            verbosity_std_row = []
            for precision in precisions:
                # Accuracy
                path = acc_dir / f"precision_{precision}" / f"predicted_labels_{prompt_id}.jsonl"
                res = evaluate_top3_predictions(path, top_k=top_k, plot=False)
                acc = res["accuracy"]
                n = res.get("total_samples", 100)  # Adjust this if needed
                se = np.sqrt(acc * (1 - acc) / n)

                accuracy_row.append(acc)
                accuracy_se_row.append(se)

                # Verbosity
                v_entry = next((v for v in verbosity_data if v['prompt_id'] == prompt_id and v['precision'] == precision), None)
                if v_entry:
                    verbosity_row.append(v_entry['mean'])
                    verbosity_std_row.append(v_entry['std'])
                else:
                    verbosity_row.append(np.nan)
                    verbosity_std_row.append(np.nan)

            accuracy_matrix.append(accuracy_row)
            accuracy_se_matrix.append(accuracy_se_row)
            verbosity_matrix.append(verbosity_row)
            verbosity_std_matrix.append(verbosity_std_row)

    return (
        np.array(accuracy_matrix), 
        np.array(verbosity_matrix),
        np.array(accuracy_se_matrix),
        np.array(verbosity_std_matrix)
)


def plot_accuracy_vs_mean(accuracy_matrix, 
                          verbosity_matrix, 
                          prompt_ids, 
                          prompt_names=None, 
                          precisions=None, 
                          top_k=3, 
                          title=None,
                          accuracy_se_matrix=None,
                          verbosity_std_matrix=None):
    """
    Plots Top-k accuracy against mean geo-term count for each prompt, with optional error bars and precision annotations.

    Args:
        accuracy_matrix (np.ndarray): 2D array [n_prompts x n_precisions] of Top-k accuracy values.
        verbosity_matrix (np.ndarray): 2D array [n_prompts x n_precisions] of mean geo-term counts.
        prompt_ids (list): List of prompt IDs used to determine line colors and order.
        prompt_names (list, optional): List of prompt names; if None, uses "Prompt <id>".
        precisions (list, optional): List of precision levels to annotate on the plot.
        top_k (int): Value of k used in Top-k accuracy (used in title).
        title (str, optional): Custom title for the plot.
        accuracy_se_matrix (np.ndarray, optional): 2D array of standard errors for accuracy values.
        verbosity_std_matrix (np.ndarray, optional): 2D array of standard deviations for geo-term counts.
    """

    plt.figure(figsize=(8, 5))
    plt.style.use("seaborn-v0_8-colorblind")

    # colormap
    cmap = get_cmap("tab20") 
    num_colors = len(prompt_ids)
    colors = [cmap(i % cmap.N) for i in range(num_colors)]

    if prompt_names is None:
        prompt_names = [f"Prompt {pid}" for pid in prompt_ids]

    for i, (acc_row, verb_row) in enumerate(zip(accuracy_matrix, verbosity_matrix)):
        label = prompt_names[i]
        color = colors[i]

        # Use error bars if provided, else None
        acc_err = accuracy_se_matrix[i] if accuracy_se_matrix is not None else None
        verb_err = verbosity_std_matrix[i] if verbosity_std_matrix is not None else None

        # Plot with error bars
        plt.errorbar(verb_row, acc_row, 
                     yerr=acc_err, xerr=verb_err,
                     fmt="o", label=label, color=color,
                     capsize=3, elinewidth=1, markeredgewidth=1)

        # Annotate with precision if provided
        if precisions is not None:
            for x, y, p in zip(verb_row, acc_row, precisions):
                plt.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)

    plt.xlabel("Mean Geo-term Count per Description", fontsize=14)
    plt.ylabel("Top-3 Prediction Accuracy", fontsize=14)
    plt.title(title or f"Accuracy vs. Geo count frequency (Top-{top_k} Predictions)", fontsize=15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 0.55)
    plt.xlim(0.0, 18.0)
    plt.grid(True)
    plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()

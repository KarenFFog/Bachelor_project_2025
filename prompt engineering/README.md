
## Experiments

- **`experiment1.py`**  
  Runs the first experimental setup.

- **`experiment2.py`**  
  Runs the second experimental setup.

- **`gen.py`, `eval.py`**  
  Helper functions used in both Experiment 1 and 2.

- **`generate.py`, `baseline.py`, `emb_image_model.py`**  
  Functions and data classes used for Experiment 3.


## Plotting

- **`plot_func.py`**  
  Contains functions to visualize experiment results.  
  Used in the notebook: `plotting.ipynb`.


## Models (`/models` folder)
Contains code for training and evaluating models for Experiment 3.

#### Training Scripts
- `run_baseline.py`  
- `pretrain.py`  
- `linear_prob.py`  
- `fine_tune.py`  

#### Evaluation Scripts
- `evaluate_baseline.py`  
- `evaluate_lin_prob.py`  
- `evaluate_ft.py`  


## Data Processing

- **`download_ben_data.py`**  
  Script to download the [BigEarthNet] dataset using the `torchgeo.datasets.BigEarthNet` class.

- **`pipeline_run.py`**  
  Generation of geographical descriptions from coordinates.

- **`split_desc_by_splits.py`**  
  Splits the generated descriptions into training, validation, and test sets.

- **`embed_descriptions.py`**  
  Embeds generated descriptions.


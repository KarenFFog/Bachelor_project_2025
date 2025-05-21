import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import os
from generate import *
from baseline import *


# load pretrained model

# remove embedding layer (final one) and insert land cover class classification layer

# freeze backbone (all layers except classifier layer)

# create subsets: 100%, 10%, 5%, 1% — stratified for multi-label - how to do that?

# train again

# evaluate on test set

# plot results (% of train used, F1 score)



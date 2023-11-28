"""
This file contains all the constants used in the project. Most of them are dictionary keys, used
for avoiding typos, making the code more readable and maintain the possibility of changing the
values without having to change every usage on the code.
"""
import os

# This repository name
REPO_NAME = 'toxicity_analysis'

# Dictionary keys
TOXIC, NON_TOXIC = 'TOXIC', 'NON-TOXIC'
TWITTER, NEWS = 'twitter', 'news-articles'

# Dataset columns
ID, TEXT, LABEL, ORIGIN = 'id','text','label','origin'
ID_TYPE, ENGLISH, FRENCH = 'id-type','english','french'

# Regex
TWITTER_REGEX = r'https://t\.co/\S+'

# Set names
TRAIN, VAL, TEST = 'TRAIN', 'VAL', 'TEST'
TRAIN_VAL = f"{TRAIN}_{VAL}"

# Visualization keys
SPANISH = 'spanish'
COLUMN_TO_LANG = {
  TEXT: SPANISH,
  ENGLISH: ENGLISH,
  FRENCH: FRENCH
}

# Internal keys for model performance analysis
TRUE_LABELS, PRED_LABELS, PRED_PROBS = 'true_labels', 'pred_labels', 'pred_probs'


# Model name
MODEL_NAME = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
MODEL_MAX_LENGTH = 512

# Experiment names
ONLY_FREEZE_EMBEDDINGS = 'only_freeze_embeddings'
FREEZE_ALL_TRANSFORMER_LAYERS_EXCEPT_LAST = 'freeze_all_transformer_layers_except_last'
FREEZE_ALL_TRANSFORMER_LAYERS = 'freeze_all_transformer_layers'

# Checkpoint Paths
CHECKPOINTS_PARENT_DIR = os.path.join('drive', 'MyDrive', 'Colab-Notebooks')
CHECKPOINT_PATHS = {
    ONLY_FREEZE_EMBEDDINGS: os.path.join(CHECKPOINTS_PARENT_DIR, ONLY_FREEZE_EMBEDDINGS),
    FREEZE_ALL_TRANSFORMER_LAYERS_EXCEPT_LAST: os.path.join(CHECKPOINTS_PARENT_DIR, FREEZE_ALL_TRANSFORMER_LAYERS_EXCEPT_LAST),
    FREEZE_ALL_TRANSFORMER_LAYERS: os.path.join(CHECKPOINTS_PARENT_DIR, FREEZE_ALL_TRANSFORMER_LAYERS)
}

# File Names
METRICS_JSON_FILE = 'metrics.json'
FULL_MODEL_LAST, MODEL_LAST = 'full_model_last.pt', 'model_last.pt'
BEST_MODEL = 'best_model.pt'


TRAIN_LOSS_HIST, TRAIN_ACC_HIST = 'train_loss_hist', 'train_acc_hist'
VAL_LOSS_HIST, VAL_ACC_HIST = 'val_loss_hist', 'val_acc_hist'
BEST_VAL_LOSS, BEST_VAL_LOSS_EPOCH = 'best_val_loss', 'best_val_loss_epoch'
ACC_AT_BEST_LOSS = 'acc_at_best_loss'

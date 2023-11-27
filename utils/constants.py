"""
This file contains all the constants used in the project. Most of them are dictionary keys, used
for avoiding typos, making the code more readable and maintain the possibility of changing the
values without having to change every usage on the code.
"""

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

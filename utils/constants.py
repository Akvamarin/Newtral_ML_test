"""
This file contains all the constants used in the project. Most of them are dictionary keys, used
for avoiding typos, making the code more readable and maintain the possibility of changing the
values without having to change every usage on the code.
"""

# Dictionary keys
TOXIC, NON_TOXIC = 'TOXIC', 'NON-TOXIC'
TWITTER, NEWS = "twitter", "news"

# Dataset columns
ID, TEXT, LABEL, ORIGIN = 'id','text','label','origin'
ID_TYPE, ENGLISH, FRENCH = 'id-type','english','french'
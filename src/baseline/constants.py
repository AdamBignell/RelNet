from pathlib import Path
import json

# local config file for main directory
with open('config.json') as my_config_file:
    config = json.load(my_config_file)

# CONSTANTS YOU NEED TO MODIFY

#whether to train on GPU
CUDA=True 
# root directory that contains the training/testing data
DATA_HOME = Path(config['DATA_HOME'])
LOG_DIR = Path(config['DATA_HOME'])
#whether to show results on the test set
PRINT_TEST = False

# CONSTANTS YOU MAY WANT TO MODIFY (BUT DON"T NEED TO)
TRAIN_DATA = DATA_HOME / "preprocessed_train_data.pkl"
VAL_DATA = DATA_HOME / "preprocessed_val_data.pkl"
TEST_DATA = DATA_HOME / "preprocessed_test_data.pkl"
BATCH_SIZE = 512
#NOTE: THESE PREPROCESSED FILES HAVE A FIXED BATCH SIZE

WORD_EMBEDS = DATA_HOME / "embeddings" / "glove_word_embeds.txt"

USER_EMBEDS = DATA_HOME / "embeddings" / "user_vecs.npy"
USER_IDS = DATA_HOME / "embeddings" / "user_vecs.vocab"

SUBREDDIT_EMBEDS = DATA_HOME / "embeddings" / "sub_vecs.npy"
SUBREDDIT_IDS = DATA_HOME / "embeddings" / "sub_vecs.vocab"

POST_INFO = DATA_HOME / "detailed_data" / "post_crosslinks_info.tsv"
LABEL_INFO = DATA_HOME / "detailed_data" / "label_info.tsv"
PREPROCESSED_DATA = DATA_HOME / "detailed_data" / "tokenized_posts.tsv"

FULL_IDS = DATA_HOME / "detailed_data" / "full_ids.txt"
FULL_EMBEDS = DATA_HOME / "detailed_data" / "full_embeds.npy"
VOCAB_SIZE = 174558
NUM_USERS = 118381
NUM_SUBREDDITS = 51278
WORD_EMBED_DIM = 300
METAFEAT_LEN = 263
NUM_CLASSES = 1
MAX_LEN = 50



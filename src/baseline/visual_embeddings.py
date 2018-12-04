import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import constants


# loads the sub embeddings and displays a 2d scatter plot
def load_embeddings():

    sub_sample = 1000 # using all the subs takes too long
    # sub reddit vectors

    sub_vecs = np.load(constants.SUBREDDIT_EMBEDS)
    vocab_file = open(constants.SUBREDDIT_IDS, "r")
    sub_ids = vocab_file.read().split()

    reduced_vec = sub_vecs[:sub_sample, :]
    sub_embed = TSNE(n_components=2).fit_transform(reduced_vec)

    plt.scatter(sub_embed[:, 0], sub_embed[:, 1], s=1)
    plt.show()


if __name__ == "__main__":
    load_embeddings()

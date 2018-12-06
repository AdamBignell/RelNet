import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import constants
import pandas as pd


def get_burst_label_info():
    label_map = {}
    source_to_dest_sub = {}
    with open(constants.LABEL_INFO) as fp:
        for line in fp:
            info = line.split("\t")
            source = info[0].split(",")[0].split("\'")[1]
            dest = info[0].split(",")[1].split("\'")[1]
            label_map[source] = 1 if info[1].strip() == "burst" else 0
    return label_map


# Create dictionary for post_crosslinks_info.tsv. Maps post ids to user, source, dest communities
def create_sub_name_dict():
    subs = []
    burst_map = get_burst_label_info()
    with open(constants.POST_INFO) as fp:
        for line in fp:
            row = line.split()

            # Values are user, source, dest
            vals = []
            post_id = row[2][0:6]
            # vals.append(post_id)
            # vals.append(row[5])
            vals.append(row[0]) # source sub name
            vals.append(row[1])  # dest sub name
            # Key is source sub name
            # vals.append(burst_map[post_id])
            if burst_map[post_id] == 1:
                subs.append(vals)

    return subs

# convert the sub name to embedding index
def sub_name_to_index(sub_names_dict, sub_sample, x, size, threshold):
    sub_id = sub_names_dict.get(x, None)
    return sub_id if sub_id is not None and sub_id < sub_sample and size >= threshold else None


def load_embeddings():

    conflict_threshold = 3  # of of conflicts before connection is drawn
    sub_sample = 300  # top n communities from the embeddings
    # sub reddit vectors

    # file with sub embedding
    sub_vecs = np.load(constants.SUBREDDIT_EMBEDS)

    # file with sub names
    vocab_file = open(constants.SUBREDDIT_IDS, "r")
    sub_names = vocab_file.read().split()
    sub_list = create_sub_name_dict()

    # get pairs of subs
    data_pairs = pd.DataFrame(sub_list, columns=['src', 'dst'])
    data_pairs = data_pairs.groupby(['src', 'dst']).size().reset_index(name='size')
    reduced_vec = sub_vecs[:sub_sample, :]
    sub_embed = TSNE(n_components=2).fit_transform(reduced_vec)

    data_embed = pd.DataFrame(sub_embed, columns=['x', 'y'])
    data_embed['src'] = pd.Series(sub_names)

    sub_names_dict = dict(zip(sub_names, range(len(sub_names))))

    data_pairs['src'] = data_pairs.apply(lambda row: sub_name_to_index(sub_names_dict, sub_sample, row['src'], row['size'], conflict_threshold), axis=1)
    data_pairs['dst'] = data_pairs.apply(lambda row: sub_name_to_index(sub_names_dict, sub_sample, row['dst'], row['size'], conflict_threshold), axis=1)
    data_pairs = data_pairs.dropna() # remove unresolved sub ids
    data_pairs['src'] = data_pairs['src'].astype('int64')
    data_pairs['dst'] = data_pairs['dst'].astype('int64')

    fig, ax = plt.subplots()

    # draw points
    ax.scatter(sub_embed[:, 0], sub_embed[:, 1], s=1)

    # draw line connections between subs
    for index, row in data_pairs.iterrows():
        ax.arrow(sub_embed[row['src']][0], sub_embed[row['src']][1], sub_embed[row['dst']][0] - sub_embed[row['src']][0], sub_embed[row['dst']][1] - sub_embed[row['src']][1], width=0.02, color='red', head_length=0.0,head_width=0.0)

    # add labels to the plot
    for i, txt in enumerate(data_embed['src'].values):
        ax.annotate(txt, (sub_embed[:, 0][i], sub_embed[:, 1][i]), fontsize=6)
    plt.show()


if __name__ == "__main__":
    load_embeddings()
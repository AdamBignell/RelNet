import numpy as np
import constants

# Take embedding index and returns post id, user, source, dest
def embed_to_txt(embed_index):
    # Load ids
    with open(constants.FULL_IDS) as fp:
        ids = {i:id.strip() for i, id in enumerate(fp.readlines())}
    
    id = ids.get(embed_index)

    # Create dictionary for post_crosslinks_info.tsv
    dict = create_dict()

    # Get user, src, dest
    entry = dict.get(id)

    return id, entry[0], entry[1], entry[2]
    

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
def create_dict():
    dict = {}
    burst_map = get_burst_label_info()
    with open(constants.POST_INFO) as fp:
        for line in fp:
            row = line.split()

            # Values are user, source, dest
            vals = []
            vals.append(row[5])
            vals.append(row[0])
            vals.append(row[1])
            # Key is post id
            key = row[2][0:6]
            vals.append(burst_map[key])
            dict[key] = vals

    return dict


# Test embed_to_txt()
def test():
    # Load the user, source, and target community embeddings for all examples
    all_embeds = np.load(open(constants.FULL_EMBEDS, 'rb'))

    print(embed_to_txt(1999))


if __name__ == "__main__":
    test()

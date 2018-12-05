import numpy as np

# SET DIRECTORY
DATA_DIR = "/Users/nadia/Desktop/RelNet/data/prediction"


# Take embedding index and returns post id, user, source, dest
def embed_to_txt(embed_index):
    # Load ids
    with open(DATA_DIR + "/detailed_data/full_ids.txt") as fp:
        ids = {i:id.strip() for i, id in enumerate(fp.readlines())}
    
    id = ids.get(embed_index)

    # Create dictionary for post_crosslinks_info.tsv
    dict = create_dict()

    # Get user, src, dest
    entry = dict.get(id)

    return id, entry[0], entry[1], entry[2]
    

# Create dictionary for post_crosslinks_info.tsv. Maps post ids to user, source, dest communities
def create_dict():
    dict = {}
    with open(DATA_DIR + "/detailed_data/post_crosslinks_info.tsv") as fp:
        for line in fp:
            row = line.split()

            # Values are user, source, dest
            vals = []
            vals.append(row[5])
            vals.append(row[0])
            vals.append(row[1])

            # Key is post id
            key = row[2][0:6]
            dict[key] = vals

    return dict


# Test embed_to_txt()
def test():
    # Load the user, source, and target community embeddings for all examples
    all_embeds = np.load(open(DATA_DIR + "/detailed_data/full_embeds.npy", 'rb'))

    print(embed_to_txt(100))


if __name__ == "__main__":
    test()

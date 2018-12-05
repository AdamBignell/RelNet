import numpy as np

# SET DIRECTORY
DATA_DIR = "/Users/nadia/Desktop/RelNet/data/prediction"

# Dest embed_to_txt
def test():
    # Load the user, source, and target community embeddings for all examples
    all_embeds = np.load(open(DATA_DIR + "/detailed_data/full_embeds.npy", 'rb'))

    #print(all_embeds[1])

    embed_to_txt(1)


# Takes embedding index and returns postid, user, source, dest
def embed_to_txt(embed_index):
    # Load ids
    with open(DATA_DIR + "/detailed_data/full_ids.txt") as fp:
        ids = {i:id.strip() for i, id in enumerate(fp.readlines())}

    create_dict()

    return ids.get(embed_index)
    

def create_dict():
    with open(DATA_DIR + "/detailed_data/post_crosslinks_info.tsv") as fp:
        rows = {i:id.strip() for i, id in enumerate(fp.readlines())}
    
    print(rows.get(1))


if __name__ == "__main__":
    test()

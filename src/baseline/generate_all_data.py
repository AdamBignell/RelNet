import numpy as np
import sys,os
import pickle

DATA_DIR = "/media/dxguo/exFAT/School/conflict_data/prediction"

def flatten(data):
    ids, text, users, subreddits, lengths,sfs, labels = [], [], [], [], [], [], []
    for batch in data:
        bids, btext, busers, bsubreddits, blengths, bsfs, blabels = batch
        ids.extend(bids)
        text.extend(btext.numpy().tolist())
        users.extend(busers.numpy().tolist())
        subreddits.extend(bsubreddits.numpy().tolist())
        lengths.extend(blengths)
        labels.extend(blabels)
        sfs.extend(bsfs)
    return (ids, text, users, subreddits, lengths, labels)

def loadFullData():
    """Loads the complete data set, largely the same as the original ipynb"""
    print("\nBegin load")

    # loading handcrafted features
    print("\tLoad handcrafted features")
    meta_features = {}
    meta_labels = {}
    numBurst = 0
    total = 0
    with open(DATA_DIR+"/detailed_data/handcrafted_features.tsv") as fp:
        for line in fp:
            info = line.split()
            meta_features[info[0]] = np.array(info[-1].split(","), dtype="f")
            meta_labels[info[0]] = 1 if info[1] == "burst" else 0
            if info[1] == "burst":
                numBurst +=1
            total += 1

    # loading the user, source, and target community embeddings for all examples
    print("\tLoad user, source, and target community embeddings")
    with open(DATA_DIR + "/detailed_data/full_ids.txt") as fp:
        ids = {id.strip():i for i, id in enumerate(fp.readlines())}
    all_embeds = np.load(open(DATA_DIR + "/detailed_data/full_embeds.npy", 'rb'))
    
    # loading the post embeddings from the LSTM 
    print("\tLoad post embeddings from the LSTM")
    lstm_embeds = np.load(open(DATA_DIR + "/detailed_data/lstm_embeds.npy", 'rb'))
    lstm_ids = pickle.load(open(DATA_DIR + "/detailed_data/lstm_embeds-ids.pkl", 'rb'))
    lstm_ids = {id:i for i, id in enumerate(lstm_ids)}

    # loading preprocessed lstm data to ensure identical train/val/test splits
    print("\tLoad preprocessed lstm data")
    train_data = pickle.load(open(DATA_DIR + "/preprocessed_train_data.pkl", 'rb'))
    val_data = pickle.load(open(DATA_DIR + "/preprocessed_val_data.pkl", 'rb'))
    test_data = pickle.load(open(DATA_DIR + "/preprocessed_test_data.pkl", 'rb'))

    print("\tFlatten data")
    flat_train_data = flatten(train_data)
    flat_val_data = flatten(val_data)
    flat_test_data = flatten(test_data)

    print("\tConcatenate data")
    train_X = np.stack([np.concatenate([meta_features[i.decode()], all_embeds[ids[i.decode()]], 
                lstm_embeds[lstm_ids[i]]]) for i in flat_train_data[0]])
    train_Y = np.stack([meta_labels[i.decode()] for i in flat_train_data[0] if i.decode() in meta_features])

    val_X = np.stack([np.concatenate([meta_features[i.decode()], all_embeds[ids[i.decode()]], 
                lstm_embeds[lstm_ids[i]]]) for i in flat_val_data[0]])
    val_Y = np.stack([meta_labels[i.decode()] for i in flat_val_data[0] if i.decode() in meta_features])

    test_X = np.stack([np.concatenate([meta_features[i.decode()], all_embeds[ids[i.decode()]], 
                lstm_embeds[lstm_ids[i]]]) for i in flat_test_data[0]])
    test_Y = np.stack([meta_labels[i.decode()] for i in flat_test_data[0] if i.decode() in meta_features])

    # np.block() stacks these different dimensional matrices
    print("\tStack data")
    all_X = np.block([[train_X], [val_X], [test_X]])    # X.shape = (x, 1227)
    all_Y = np.block([train_Y, val_Y, test_Y])          # Y.shape = (y,)

    prop_all = []
    for i in range(len(all_X)):
        prop_all.append((all_X[i], all_Y[i]))

    print("\tSave data")
    np.save(DATA_DIR + '/prop_subsample.npy', prop_all[0:10])
    np.save(DATA_DIR + '/prop_all.npy', prop_all)

    print("\nFinished saving data!")
    return


def main():
    loadFullData()

if __name__ == "__main__":
    main()
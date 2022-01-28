import argparse
import numpy as np

# Arguments
parser = argparse.ArgumentParser(description="Generate training network")
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument("--weighted", action="store_true")
args = parser.parse_args()


# Argument Parse
dataset = args.dataset
train_filename = "train_%s.edgelist" % dataset
train_weighted_filename = "train_%s_weighted.edgelist" % dataset
emb_filename = "%s-struc2vec.emb" % dataset


# Load training edgelist
with open("split/" + train_filename, "r") as f:
    lines = f.readlines()

pairs = list()

for line in lines:
    f, t = line.strip("\n").split()
    pair = [int(f), int(t)]
    pairs.append(pair)


# Write training edgelist
with open("split/" + train_weighted_filename, "w") as f:
    if args.weighted:
        try:
            with open("emb/" + emb_filename, "r") as g:
                _ = g.readline()
                lines = g.readlines()

            node_embedding = dict()

            for line in lines:
                node_id, emb = line.strip("\n").split(" ", 1)
                node_id = int(node_id)
                emb = emb.split()
                emb = [float(e) for e in emb]
                node_embedding[node_id] = np.array(emb)
            
            for pair in pairs:
                f.write("%d %d %.8f\n" % (pair[0], pair[1], np.dot(node_embedding[pair[0]], node_embedding[pair[1]])))

        except:
            for pair in pairs:
                f.write("%d %d 1\n" % (pair[0], pair[1]))
    else:
        for pair in pairs:
            f.write("%d %d 1\n" % (pair[0], pair[1]))

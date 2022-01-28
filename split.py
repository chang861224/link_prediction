import argparse
import numpy as np

# Arguments
parser = argparse.ArgumentParser(description="Link prediction task")
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument("--train_percent", type=float, default=85.0)
parser.add_argument("--valid_percent", type=float, default=5.0)
parser.add_argument("--test_percent", type=float, default=10.0)
args = parser.parse_args()


# Argument Parse
dataset = args.dataset
dataset_path = "graph/%s.edgelist" % dataset
train_filename = "train_%s.edgelist" % dataset
valid_filename = "valid_%s.edgelist" % dataset
test_filename = "test_%s.edgelist" % dataset
train_percent = args.train_percent
valid_percent = args.valid_percent
test_percent = args.test_percent


# Edge Pre-Processing
with open(dataset_path, "r") as f:
    lines = f.readlines()

pos = dict()
neg = dict()

for idx, line in enumerate(lines):
    source, target = line.strip("\n").split()
    source = int(source)
    target = int(target)
    
    try:
        pos[source].append(target)
    except:
        pos[source] = list()
        pos[source].append(target)

all_nodes = set(list(pos.keys()))

for key in pos.keys():
    nodes = np.random.permutation(np.array(list(all_nodes - set(pos[key]) - set([key])))).tolist()[:len(pos[key])]
    neg[key] = nodes

pos_edges = list()
neg_edges = list()

for key in pos.keys():
    for target in pos[key]:
        pos_edges.append([key, target])

    for target in neg[key]:
        neg_edges.append([key, target])


# Random Permutation
pos_edges = np.array(pos_edges)
pos_edges = np.random.permutation(pos_edges)
neg_edges = np.array(neg_edges)
neg_edges = np.random.permutation(neg_edges)


# Number of training, validation, and testing edges
num_train = int(len(pos_edges) * train_percent / 100)
num_valid = int(len(pos_edges) * valid_percent / 100)
num_test = int(len(pos_edges) * test_percent / 100)


# Generate training, validation, and testing edges
train_pos = pos_edges.tolist()[:num_train]
valid_pos = pos_edges.tolist()[num_train:num_train + num_valid]
valid_neg = neg_edges.tolist()[num_train:num_train + num_valid]
test_pos = pos_edges.tolist()[-num_test:]
test_neg = neg_edges.tolist()[-num_test:]


# Write training edges to file
with open("split/" + train_filename, "w") as f:
    for pair in train_pos:
        f.write("%d %d\n" % (pair[0], pair[1]))


# Write validation edges to file
with open("split/" + valid_filename, "w") as f:
    f.write("%d %d\n" % (len(valid_pos), len(valid_neg)))

    for pair in valid_pos:
        f.write("%d %d\n" % (pair[0], pair[1]))
    for pair in valid_neg:
        f.write("%d %d\n" % (pair[0], pair[1]))


# Write testing edges to file
with open("split/" + test_filename, "w") as f:
    f.write("%d %d\n" % (len(test_pos), len(test_neg)))

    for pair in test_pos:
        f.write("%d %d\n" % (pair[0], pair[1]))
    for pair in test_neg:
        f.write("%d %d\n" % (pair[0], pair[1]))

import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser(description="Link prediction task")
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument("--embed", type=str, default="emb/CiteSeer-struc2vec.emb")
parser.add_argument("--test_percent", type=float, default=1.0)
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()

dataset_path = "graph/%s.edgelist" % args.dataset
embed_path = args.embed
test_perent = args.test_percent
runs = args.runs

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

with open(embed_path, "r") as f:
    f.readline()
    lines = f.readlines()

embeddings = dict()

for line in lines:
    node, embedding = line.strip("\n").split(" ", 1)
    node = int(node)
    embedding = embedding.split()
    embedding = [float(e) for e in embedding]
    embeddings[node] = np.array(embedding)

scores = list()

for _ in range(runs):
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

    num_nodes = int(len(pos_edges) * test_perent / 100)

    test_pos = np.random.permutation(np.array(pos_edges)).tolist()[:num_nodes]
    test_neg = np.random.permutation(np.array(neg_edges)).tolist()[:num_nodes]

    y_true = np.array([1] * num_nodes + [0] * num_nodes)
    y_pred = list()

    for pair in (test_pos + test_neg):
        score = np.dot(embeddings[pair[0]], embeddings[pair[1]])
        y_pred.append(score)

    y_pred = np.array(y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    scores.append(auc(fpr, tpr))

scores = np.array(scores)
print("AUC Score: %.4f, %.4f" % (np.average(scores), np.std(scores)))

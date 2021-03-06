import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Arguments
parser = argparse.ArgumentParser(description="Link prediction task")
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument("--embed", type=str, default="emb/CiteSeer-struc2vec.emb")
args = parser.parse_args()


# Argument Parse
dataset = args.dataset
embed_path = args.embed


# Load node embedding
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


# Predict validation data
with open("split/valid_%s.edgelist" % dataset, "r") as f:
    num_pos, num_neg = f.readline().strip("\n").split()
    num_pos = int(num_pos)
    num_neg = int(num_neg)
    lines = f.readlines()

edges = list()

for line in lines:
    source, target = line.strip("\n").split()
    source = int(source)
    target = int(target)
    edges.append([source, target])

y_true = np.array([1] * num_pos + [0] * num_neg)
y_pred = list()

for pair in edges:
    try:
        score = np.dot(embeddings[pair[0]], embeddings[pair[1]])
        y_pred.append(score)
    except:
        y_pred.append(0)

y_pred = np.array(y_pred).reshape(-1, 1)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(y_pred, y_true)
print("Validation AUC Score: %.4f" % roc_auc_score(y_true, clf.decision_function(y_pred)))


# Predict testing data
with open("split/test_%s.edgelist" % dataset, "r") as f:
    num_pos, num_neg = f.readline().strip("\n").split()
    num_pos = int(num_pos)
    num_neg = int(num_neg)
    lines = f.readlines()

edges = list()

for line in lines:
    source, target = line.strip("\n").split()
    source = int(source)
    target = int(target)
    edges.append([source, target])

y_true = np.array([1] * num_pos + [0] * num_neg)
y_pred = list()

for pair in edges:
    try:
        score = np.dot(embeddings[pair[0]], embeddings[pair[1]])
        y_pred.append(score)
    except:
        y_pred.append(0)

y_pred = np.array(y_pred).reshape(-1, 1)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(y_pred, y_true)
print("Testing AUC Score: %.4f" % roc_auc_score(y_true, clf.decision_function(y_pred)))

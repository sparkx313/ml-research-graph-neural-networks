# Sparse Graph Convolutional Node Classifier

Train a two-layer graph convolutional network (GCN) on a graph dataset you provide. The implementation reads node features, optional node labels, and edges from CSV; builds a sparse, normalized adjacency matrix; trains with cross-entropy and Adam; reports a stratified label holdout; saves model weights; and writes predictions for every node.

It trains on supplied data. There are no generated labels, placeholder model scores, or bundled sample datasets.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Input files

The node CSV has one row per node. It must contain a unique `node_id`, a `label` column, and one or more numeric feature columns. Leave `label` empty for nodes you want predicted but do not include in the supervised split.

```text
node_id,label,feature_1,feature_2,...
```

The edge CSV contains `source,target` node IDs. Edges are treated as undirected, duplicate edges are collapsed, and self-loops are added by the GCN preprocessing step.

```text
source,target
```

## Train and predict

```bash
python main.py train \
  --nodes data/nodes.csv \
  --edges data/edges.csv \
  --model-out artifacts/gcn_model.npz \
  --predictions-out reports/predictions.csv
```

Training prints the best validation epoch, holdout cross-entropy, accuracy, and macro F1. The model archive stores its weights, class labels, feature-column order, and feature scaling parameters.

Apply a saved model to another graph with the same feature columns in the same order:

```bash
python main.py predict \
  --nodes new_graph/nodes.csv \
  --edges new_graph/edges.csv \
  --model artifacts/gcn_model.npz \
  --output reports/new_graph_predictions.csv
```

Predictions contain each node ID, its highest-scoring class, and that class's softmax confidence.

## Model and evaluation scope

- Uses a sparse symmetric normalized adjacency matrix, a ReLU hidden layer, and a softmax output layer. No dense node-by-node adjacency matrix is created.
- The random seed, validation fraction, epoch limit, early-stopping patience, learning rate, hidden size, and L2 weight decay can be set on the command line.
- The validation split withholds labels only. All node features and graph edges remain available during message passing, so this is a transductive evaluation. For a strict inductive benchmark, supply a split and graph construction designed for that protocol.
- Results depend on the supplied graph and labels. This repository does not claim published benchmark scores and does not implement GAT, GraphSAGE, molecular message passing, or graph-level property prediction.

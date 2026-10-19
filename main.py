import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Graph:
    n_nodes: int
    edge_list: List[Tuple[int, int]]
    node_features: np.ndarray
    node_labels: Optional[np.ndarray] = None

    def normalized_adj(self) -> np.ndarray:
        A = np.zeros((self.n_nodes, self.n_nodes))
        for u, v in self.edge_list:
            A[u,v] = A[v,u] = 1.0
        A = A + np.eye(self.n_nodes)
        D = np.diag(A.sum(axis=1)**-0.5)
        return D @ A @ D

class GCNLayer:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim)
    def forward(self, X, A_norm): return np.maximum(0, A_norm @ X @ self.W)

class GNN:
    def __init__(self, in_dim, hidden, n_classes):
        self.l1 = GCNLayer(in_dim, hidden)
        self.l2 = GCNLayer(hidden, n_classes)

    def forward(self, graph: Graph) -> np.ndarray:
        A = graph.normalized_adj()
        h = self.l1.forward(graph.node_features, A)
        out = self.l2.forward(h, A)
        exp = np.exp(out - out.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

def generate_graph(n_nodes=500, n_features=100, n_classes=7) -> Graph:
    np.random.seed(42)
    features = (np.random.rand(n_nodes, n_features) > 0.95).astype(np.float32)
    edges = []
    for i in range(n_nodes):
        for _ in range(np.random.randint(1, 6)):
            j = np.random.randint(0, n_nodes)
            if j != i: edges.append((i,j))
    labels = np.random.randint(0, n_classes, n_nodes)
    return Graph(n_nodes, edges, features, labels)

if __name__ == "__main__":
    graph = generate_graph(500, 100, 7)
    print(f"Graph: {graph.n_nodes} nodes, {len(graph.edge_list)} edges")
    model = GNN(100, 64, 7)
    preds = model.forward(graph)
    pred_classes = np.argmax(preds, axis=1)
    acc = np.mean(pred_classes == graph.node_labels)
    print(f"GCN output shape: {preds.shape}, random baseline acc: {acc:.3f}")

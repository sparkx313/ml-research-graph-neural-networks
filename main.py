"""Train and apply a sparse, two-layer graph convolutional node classifier."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def read_graph(nodes_path: Path, edges_path: Path) -> tuple[list[str], list[str], np.ndarray, sparse.csr_matrix]:
    if not nodes_path.is_file():
        raise ValueError(f"Node CSV does not exist: {nodes_path}")
    if not edges_path.is_file():
        raise ValueError(f"Edge CSV does not exist: {edges_path}")

    node_ids: list[str] = []
    labels: list[str] = []
    features: list[list[float]] = []
    with nodes_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Node CSV is missing a header row.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if any(not name for name in reader.fieldnames):
            raise ValueError("Node CSV column names cannot be empty.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Node CSV column names must be unique.")
        if "node_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Node CSV must contain node_id and label columns.")
        feature_columns = [name for name in reader.fieldnames if name not in {"node_id", "label"}]
        if not feature_columns:
            raise ValueError("Node CSV must contain at least one numeric feature column.")

        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Node CSV row {row_number} has more values than the header.")
            node_id = (row.get("node_id") or "").strip()
            if not node_id:
                raise ValueError(f"Node CSV row {row_number} is missing node_id.")
            if node_id in seen_ids:
                raise ValueError(f"Node CSV row {row_number} duplicates node_id {node_id!r}.")
            seen_ids.add(node_id)
            label = (row.get("label") or "").strip()
            values: list[float] = []
            for column in feature_columns:
                try:
                    value = float(row.get(column, ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Node CSV row {row_number}: feature {column!r} must be numeric."
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"Node CSV row {row_number}: feature {column!r} must be finite.")
                values.append(value)
            node_ids.append(node_id)
            labels.append(label)
            features.append(values)

    if len(node_ids) < 2:
        raise ValueError("Node CSV must contain at least two rows.")
    feature_matrix = np.asarray(features, dtype=np.float32)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature values exceed the supported numeric range.")
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    rows: list[int] = []
    columns: list[int] = []
    with edges_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Edge CSV is missing a header row.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if any(not name for name in reader.fieldnames):
            raise ValueError("Edge CSV column names cannot be empty.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Edge CSV column names must be unique.")
        if not {"source", "target"}.issubset(reader.fieldnames):
            raise ValueError("Edge CSV must contain source and target columns.")
        for row_number, edge in enumerate(reader, start=2):
            if None in edge:
                raise ValueError(f"Edge CSV row {row_number} has more values than the header.")
            source_id = (edge.get("source") or "").strip()
            target_id = (edge.get("target") or "").strip()
            if source_id not in node_index or target_id not in node_index:
                raise ValueError(f"Edge CSV row {row_number} references an unknown node.")
            source_index = node_index[source_id]
            target_index = node_index[target_id]
            rows.extend((source_index, target_index))
            columns.extend((target_index, source_index))

    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency + sparse.eye(len(node_ids), dtype=np.float32, format="csr")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.asarray(np.power(degrees, -0.5), dtype=np.float32)
    scaling = sparse.diags(inverse_sqrt)
    normalized_adjacency = (scaling @ adjacency @ scaling).tocsr()
    return node_ids, labels, feature_matrix, normalized_adjacency


def stratified_split(
    labeled_indices: np.ndarray,
    encoded_labels: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    validation: list[int] = []
    for label in np.unique(encoded_labels[labeled_indices]):
        class_indices = labeled_indices[encoded_labels[labeled_indices] == label].copy()
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            train.extend(class_indices.tolist())
            continue
        count = max(1, int(round(len(class_indices) * validation_fraction)))
        count = min(count, len(class_indices) - 1)
        validation.extend(class_indices[:count].tolist())
        train.extend(class_indices[count:].tolist())
    if not train or not validation:
        raise ValueError("Need at least two labeled nodes in one class to create a holdout split.")
    rng.shuffle(train)
    rng.shuffle(validation)
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def probabilities(
    features: np.ndarray, adjacency: sparse.csr_matrix, weights_1: np.ndarray, weights_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support_1 = adjacency @ features
    hidden_linear = support_1 @ weights_1
    hidden = np.maximum(hidden_linear, 0)
    support_2 = adjacency @ hidden
    logits = support_2 @ weights_2
    return softmax(logits), support_1, hidden_linear


def cross_entropy(probabilities_: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> float:
    selected = np.clip(probabilities_[indices, labels[indices]], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def train_gcn(
    features: np.ndarray,
    adjacency: sparse.csr_matrix,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    hidden_dim: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    feature_mean = features[train_indices].mean(axis=0)
    feature_scale = features[train_indices].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized_features = (features - feature_mean) / feature_scale

    input_dim = features.shape[1]
    class_count = int(labels.max()) + 1
    limit_1 = np.sqrt(6.0 / (input_dim + hidden_dim))
    limit_2 = np.sqrt(6.0 / (hidden_dim + class_count))
    weights_1 = rng.uniform(-limit_1, limit_1, (input_dim, hidden_dim)).astype(np.float32)
    weights_2 = rng.uniform(-limit_2, limit_2, (hidden_dim, class_count)).astype(np.float32)

    moment_1 = np.zeros_like(weights_1)
    variance_1 = np.zeros_like(weights_1)
    moment_2 = np.zeros_like(weights_2)
    variance_2 = np.zeros_like(weights_2)
    best_loss = float("inf")
    best_weights_1, best_weights_2 = weights_1.copy(), weights_2.copy()
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        probabilities_, support_1, hidden_linear = probabilities(
            normalized_features, adjacency, weights_1, weights_2
        )
        grad_logits = np.zeros_like(probabilities_)
        grad_logits[train_indices] = probabilities_[train_indices]
        grad_logits[train_indices, labels[train_indices]] -= 1.0
        grad_logits[train_indices] /= len(train_indices)

        grad_weights_2 = (adjacency @ np.maximum(hidden_linear, 0)).T @ grad_logits
        grad_hidden = adjacency.T @ (grad_logits @ weights_2.T)
        grad_hidden_linear = grad_hidden * (hidden_linear > 0)
        grad_weights_1 = support_1.T @ grad_hidden_linear
        grad_weights_1 += weight_decay * weights_1
        grad_weights_2 += weight_decay * weights_2

        beta_1, beta_2 = 0.9, 0.999
        moment_1 = beta_1 * moment_1 + (1 - beta_1) * grad_weights_1
        variance_1 = beta_2 * variance_1 + (1 - beta_2) * np.square(grad_weights_1)
        moment_2 = beta_1 * moment_2 + (1 - beta_1) * grad_weights_2
        variance_2 = beta_2 * variance_2 + (1 - beta_2) * np.square(grad_weights_2)
        correction_1 = 1 - beta_1**epoch
        correction_2 = 1 - beta_2**epoch
        weights_1 -= learning_rate * (moment_1 / correction_1) / (
            np.sqrt(variance_1 / correction_2) + 1e-8
        )
        weights_2 -= learning_rate * (moment_2 / correction_1) / (
            np.sqrt(variance_2 / correction_2) + 1e-8
        )

        validation_probabilities, _, _ = probabilities(
            normalized_features, adjacency, weights_1, weights_2
        )
        validation_loss = cross_entropy(validation_probabilities, labels, validation_indices)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_weights_1, best_weights_2 = weights_1.copy(), weights_2.copy()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_weights_1, best_weights_2, best_loss, best_epoch


def macro_f1(truth: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    scores: list[float] = []
    for label in range(class_count):
        true_positive = int(np.sum((truth == label) & (predicted == label)))
        false_positive = int(np.sum((truth != label) & (predicted == label)))
        false_negative = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def write_predictions(path: Path, node_ids: list[str], classes: list[str], scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("node_id", "predicted_label", "confidence"))
        for node_id, row in zip(node_ids, scores):
            class_index = int(np.argmax(row))
            writer.writerow((node_id, classes[class_index], f"{row[class_index]:.8f}"))


def save_model(
    path: Path,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    classes: list[str],
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError("Model output filename must end in .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights_1=weights_1,
        weights_2=weights_2,
        classes=np.asarray(classes, dtype=np.str_),
        feature_columns=np.asarray(feature_columns, dtype=np.str_),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def load_model(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Model file does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name].copy() for name in saved.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read model archive: {exc}") from exc


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.")
    return parsed


def fraction(value: str) -> float:
    parsed = positive_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("Validation fraction must be below 1.")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sparse GCN node classifier on a supplied graph.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train, evaluate on a holdout, save, and predict.")
    train.add_argument("--nodes", required=True, type=Path, help="CSV with node_id,label,and numeric feature columns.")
    train.add_argument("--edges", required=True, type=Path, help="CSV with source,target node IDs; edges are undirected.")
    train.add_argument("--model-out", type=Path, default=Path("gcn_model.npz"))
    train.add_argument("--predictions-out", type=Path, default=Path("predictions.csv"))
    train.add_argument("--hidden-dim", type=positive_int, default=64)
    train.add_argument("--epochs", type=positive_int, default=200)
    train.add_argument("--patience", type=positive_int, default=30)
    train.add_argument("--learning-rate", type=positive_float, default=0.01)
    train.add_argument("--weight-decay", type=nonnegative_float, default=0.0005)
    train.add_argument("--validation-fraction", type=fraction, default=0.2)
    train.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser("predict", help="Apply a saved model to nodes in a supplied graph.")
    predict.add_argument("--nodes", required=True, type=Path)
    predict.add_argument("--edges", required=True, type=Path)
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def run_train(args: argparse.Namespace) -> None:
    node_ids, labels, features, adjacency = read_graph(args.nodes, args.edges)
    classes = sorted({label for label in labels if label})
    if len(classes) < 2:
        raise ValueError("Training needs at least two distinct non-empty labels.")
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_index.get(label, -1) for label in labels], dtype=np.int64)
    labeled_indices = np.flatnonzero(encoded >= 0)
    if len(labeled_indices) < 3:
        raise ValueError("Training needs at least three labeled nodes.")
    rng = np.random.default_rng(args.seed)
    train_indices, validation_indices = stratified_split(
        labeled_indices, encoded, args.validation_fraction, rng
    )
    weights_1, weights_2, validation_loss, best_epoch = train_gcn(
        features,
        adjacency,
        encoded,
        train_indices,
        validation_indices,
        args.hidden_dim,
        args.epochs,
        args.patience,
        args.learning_rate,
        args.weight_decay,
        rng,
    )
    feature_mean = features[train_indices].mean(axis=0)
    feature_scale = features[train_indices].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    model_columns = [f"feature_{index}" for index in range(features.shape[1])]
    # Preserve the actual CSV feature names so prediction can verify column order.
    with args.nodes.open("r", encoding="utf-8-sig", newline="") as source:
        model_columns = [name.strip() for name in next(csv.reader(source)) if name.strip() not in {"node_id", "label"}]
    save_model(
        args.model_out, weights_1, weights_2, classes, model_columns, feature_mean, feature_scale
    )
    scores, _, _ = probabilities(
        (features - feature_mean) / feature_scale, adjacency, weights_1, weights_2
    )
    write_predictions(args.predictions_out, node_ids, classes, scores)
    validation_predictions = np.argmax(scores[validation_indices], axis=1)
    validation_truth = encoded[validation_indices]
    print(f"Best epoch: {best_epoch}; holdout cross-entropy: {validation_loss:.6f}")
    print(
        f"Holdout accuracy: {np.mean(validation_predictions == validation_truth):.4f}; "
        f"macro F1: {macro_f1(validation_truth, validation_predictions, len(classes)):.4f}"
    )
    print(f"Saved model to {args.model_out} and predictions to {args.predictions_out}.")


def run_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    node_ids, _labels, features, adjacency = read_graph(args.nodes, args.edges)
    feature_columns = model["feature_columns"].astype(str).tolist()
    with args.nodes.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    supplied_columns = [name.strip() for name in header if name.strip() not in {"node_id", "label"}]
    if supplied_columns != feature_columns:
        raise ValueError("Prediction feature columns must match the model's training columns and order.")
    if features.shape[1] != len(model["feature_mean"]):
        raise ValueError("Prediction feature count does not match the saved model.")
    normalized_features = (features - model["feature_mean"]) / model["feature_scale"]
    scores, _, _ = probabilities(
        normalized_features, adjacency, model["weights_1"], model["weights_2"]
    )
    classes = model["classes"].astype(str).tolist()
    if scores.shape[1] != len(classes):
        raise ValueError("Saved model class metadata does not match its output weights.")
    write_predictions(args.output, node_ids, classes, scores)
    print(f"Predicted labels for {len(node_ids)} nodes; wrote {args.output}.")


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_paths = {args.nodes.resolve(), args.edges.resolve()}
        output_paths = (
            {args.model_out.resolve(), args.predictions_out.resolve()}
            if args.command == "train"
            else {args.output.resolve()}
        )
        if len(output_paths) < (2 if args.command == "train" else 1):
            raise ValueError("Model and prediction outputs must use different paths.")
        if input_paths & output_paths:
            raise ValueError("Output paths must not overwrite the node or edge CSV inputs.")
        if args.command == "train":
            run_train(args)
        else:
            if args.model.resolve() == args.output.resolve():
                raise ValueError("Prediction output must not overwrite the model archive.")
            run_predict(args)
    except (OSError, ValueError, csv.Error, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import prepare_dataset
from tmu.tools import BenchmarkTimer
import pandas as pd
import os
import numpy as np

# -------------------------
# Graph Construction
# -------------------------
def build_graph(x_row, y_label, bits=16):
    u, i, c = map(int, x_row)

    def bin_encode(v):
        return torch.tensor([(v >> b) & 1 for b in range(bits)], dtype=torch.float)

    x = torch.stack([
        torch.cat([bin_encode(u), torch.zeros(bits * 2)]),
        torch.cat([torch.zeros(bits), bin_encode(i), torch.zeros(bits)]),
        torch.cat([torch.zeros(bits * 2), bin_encode(c)]),
    ])

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ], dtype=torch.long)

    y = torch.tensor(y_label, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# -------------------------
# GNN Model (Graph-level)
# -------------------------
class GraphClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # graph-level embedding
        return self.classifier(x)

# -------------------------
# Main Experiment
# -------------------------
def main(args):
    results = []

    data = prepare_dataset.aug_amazon_products(
        noise_ratio=args.dataset_noise_ratio
    )
    x, y = prepare_dataset.construct_x_y(data)
    X_train, X_test, Y_train, Y_test = prepare_dataset.train_test_split(x, y)

    bits = 16
    feature_dim = 3 * bits
    # Build graph datasets
    train_graphs = [build_graph(X_train[i], Y_train[i], bits=bits) for i in range(len(X_train))]
    test_graphs = [build_graph(X_test[i], Y_test[i], bits=bits) for i in range(len(X_test))]

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64)

    num_classes = len(np.unique(y))

    model = GraphClassifier(
        in_dim=feature_dim,
        hidden_dim=64,
        num_classes=num_classes
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    benchmark_total = BenchmarkTimer(logger=None, text="Total Time")
    with benchmark_total:
        for epoch in range(args.epochs):
            # Training
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()

            # Testing
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in test_loader:
                        out = model(batch.x, batch.edge_index, batch.batch)
                        pred = out.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        total += batch.y.size(0)

                accuracy = 100.0 * correct / total

    total_time = benchmark_total.elapsed()

    results.append({
        "Exp_id": args.exp_id,
        "Algorithm": "Graph NN",
        "Noise_Ratio": args.dataset_noise_ratio,
        "T": 0,
        "s": 0,
        "Max_Included_Literals": 0,
        "Epochs": args.epochs,
        "Platform": args.platform,
        "Total_Time": total_time,
        "Accuracy": accuracy,
    })

    results_df = pd.DataFrame(results)
    results_file = "experiment_results.csv"
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", index=False, header=False)
    else:
        results_df.to_csv(results_file, index=False)

    print(f"Results saved to {results_file}")

# -------------------------
# Arguments
# -------------------------
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--dataset_noise_ratio", default=0.01, type=float)
    parser.add_argument("--platform", default="GPU", type=str)
    parser.add_argument("--exp_id", default="", type=str)
    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args

if __name__ == "__main__":
    main(default_args())
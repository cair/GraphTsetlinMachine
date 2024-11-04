from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import random
import pandas as pd
import kagglehub

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=32, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--max-included-literals", default=4, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

print("Creating training data")
path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
print("Path to dataset files:", path)
data_file = path + "/Reviews.csv"  # Adjust this path if necessary
data = pd.read_csv(data_file)
print("Data preview:", data.head())

number_of_nodes = 3

symbols = []
users = data['user_id'].unique()
items = data['product_id'].unique()
categories = data['category'].unique()

# Initialize Graphs with symbols for GTM
num_graphs = len(items)
symbols = ["I" + str(i) for i in items] + ["C" + str(c) for c in categories] + ["U" + str(u) for u in users]

graphs_train = Graphs(
    X_train.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import prepare_dataset

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=4096, type=int)
    parser.add_argument("--hypervector-bits", default=256, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--max-included-literals", default=3, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args
args = default_args()
np.random.seed(42)

# data = prepare_dataset.amazon_products()
data = prepare_dataset.aug_amazon_products()
# data = prepare_dataset.artificial()
# data = prepare_dataset.artificial_with_user_pref()
# data = prepare_dataset.artificial_pattered()
# print(data.head())
le_user = LabelEncoder()
le_item = LabelEncoder()
le_category = LabelEncoder()
le_rating = LabelEncoder() 
data['user_id'] = le_user.fit_transform(data['user_id'])
data['product_id'] = le_item.fit_transform(data['product_id'])
data['category'] = le_category.fit_transform(data['category'])
data['rating'] = le_rating.fit_transform(data['rating'])
x = data[['user_id', 'product_id', 'category']].values  
y = data['rating'].values 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", Y_test.shape)
users = data['user_id'].unique()
items = data['product_id'].unique()
categories = data['category'].unique()
# Initialize Graphs with symbols for GTM
number_of_nodes = 3
symbols = []
symbols = ["U_" + str(u) for u in users] + ["I_" + str(i) for i in items] + ["C_" + str(c) for c in categories] 
print("Symbols: ",len(symbols))

# Train data
graphs_train = Graphs(
    X_train.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)
for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)
graphs_train.prepare_node_configuration()
for graph_id in range(X_train.shape[0]):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_train.number_of_graph_nodes[graph_id]-1 else 1
        if node_id == 0:
            graphs_train.add_graph_node(graph_id, "User", number_of_edges)
        elif node_id == 1:
            graphs_train.add_graph_node(graph_id, "Item", number_of_edges)
        else:
            graphs_train.add_graph_node(graph_id, "Category", number_of_edges)
graphs_train.prepare_edge_configuration()
for graph_id in range(X_train.shape[0]):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        if node_id == 0:
            graphs_train.add_graph_node_edge(graph_id, "User", "Item", "UserItem")
            
        if node_id == 1:
            graphs_train.add_graph_node_edge(graph_id, "Item", "Category", "ItemCategory")
            graphs_train.add_graph_node_edge(graph_id, "Item", "User", "ItemUser")
            
        if node_id == 2:
            graphs_train.add_graph_node_edge(graph_id, "Category", "Item", "CatrgoryItem")

    graphs_train.add_graph_node_property(graph_id, "User", "U_" + str(X_train[graph_id][0]))
    graphs_train.add_graph_node_property(graph_id, "Item", "I_" + str(X_train[graph_id][1]))
    graphs_train.add_graph_node_property(graph_id, "Category", "C_" + str(X_train[graph_id][2]))
graphs_train.encode()
print("Training data produced")

# Test data
graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)
graphs_test.prepare_node_configuration()
for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        number_of_edges = 2 if node_id > 0 and node_id < graphs_test.number_of_graph_nodes[graph_id]-1 else 1
        if node_id == 0:
            graphs_test.add_graph_node(graph_id, "User", number_of_edges)
        elif node_id == 1:
            graphs_test.add_graph_node(graph_id, "Item", number_of_edges)
        else:
            graphs_test.add_graph_node(graph_id, "Category", number_of_edges)
graphs_test.prepare_edge_configuration()
for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        if node_id == 0:
            graphs_test.add_graph_node_edge(graph_id, "User", "Item", "UserItem")
            
        if node_id == 1:
            graphs_test.add_graph_node_edge(graph_id, "Item", "Category", "ItemCategory")
            graphs_test.add_graph_node_edge(graph_id, "Item", "User", "ItemUser")
            
        if node_id == 2:
            graphs_test.add_graph_node_edge(graph_id, "Category", "Item", "CatrgoryItem")

    graphs_test.add_graph_node_property(graph_id, "User", "U_" + str(X_test[graph_id][0]))
    graphs_test.add_graph_node_property(graph_id, "Item", "I_" + str(X_test[graph_id][1]))
    graphs_test.add_graph_node_property(graph_id, "Category", "C_" + str(X_test[graph_id][2]))
graphs_test.encode()
print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    double_hashing = args.double_hashing
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

# weights = tm.get_state()[1].reshape(2, -1)
# for i in range(tm.number_of_clauses):
#         print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
#         l = []
#         for k in range(args.hypervector_size * 2):
#             if tm.ta_action(0, i, k):
#                 if k < args.hypervector_size:
#                     l.append("x%d" % (k))
#                 else:
#                     l.append("NOT x%d" % (k - args.hypervector_size))

#         for k in range(args.message_size * 2):
#             if tm.ta_action(1, i, k):
#                 if k < args.message_size:
#                     l.append("c%d" % (k))
#                 else:
#                     l.append("NOT c%d" % (k - args.message_size))

#         print(" AND ".join(l))

# print(graphs_test.hypervectors)
# print(tm.hypervectors)
# print(graphs_test.edge_type_id)
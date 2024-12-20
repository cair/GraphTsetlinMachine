from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=4096, type=int)
    parser.add_argument("--hypervector-bits", default=256, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--max-included-literals", default=10, type=int)
    parser.add_argument("--number-of-examples", default=1000, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

############################# real dataset ########################

print("Creating training data")
path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
print("Path to dataset files:", path)
data_file = path + "/amazon.csv" 
org_data = pd.read_csv(data_file)
# print("Data preview:", data.head())
org_data = org_data[['product_id', 'category', 'user_id', 'rating']]
#################################### expanded 
org_data['rating'] = pd.to_numeric(org_data['rating'], errors='coerce')  # Coerce invalid values to NaN
org_data.dropna(subset=['rating'], inplace=True)  # Drop rows with NaN ratings
org_data['rating'] = org_data['rating'].astype(int)
# Expand the dataset 10 times
data = pd.concat([org_data] * 10, ignore_index=True)

# Shuffle the expanded dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Add noise
# Define the noise ratio
noise_ratio = 0.1  # 10% noise

# Select rows to apply noise
num_noisy_rows = int(noise_ratio * len(data))
noisy_indices = np.random.choice(data.index, size=num_noisy_rows, replace=False)

# Add noise to ratings
data.loc[noisy_indices, 'rating'] = np.random.choice(range(1, 6), size=num_noisy_rows)

# Add noise to categories
unique_categories = data['category'].unique()
data.loc[noisy_indices, 'category'] = np.random.choice(unique_categories, size=num_noisy_rows)

# Print a preview of the noisy and expanded dataset
print("Original data shape:", org_data.shape)
print("Expanded data shape:", data.shape)
print("Data preview:\n", data.head())
############################# artificial dataset ########################

# Set random seed for reproducibility
# np.random.seed(42)

########################## ver 1 ############################

# num_users = 5  # Number of unique users
# num_items =10  # Number of unique items
# num_categories = 5  # Number of unique categories
# num_interactions = 1000  # Number of user-item interactions
# # Generate random ratings (e.g., between 1 and 5)
# ratings = np.random.choice(range(1, 3), num_interactions)
# # Generate random user-item interactions
# user_ids = np.random.choice(range(num_users), num_interactions)
# item_ids = np.random.choice(range(num_items), num_interactions)
# categories = np.random.choice(range(num_categories), num_interactions)

# data = pd.DataFrame({
#     'user_id': user_ids,
#     'product_id': item_ids,
#     'category': categories,
#     'rating': ratings
# })
# print("Artificial Dataset Preview:")

########################## ver 2 ############################

# Parameters
# num_users = 100  # Number of unique users
# num_items = 50    # Number of unique items
# num_categories = 50  # Number of unique categories
# num_interactions = 1000  # Number of user-item interactions
# noise_ratio = 0.01  # Percentage of noisy interactions

# # Generate user preferences: each user prefers 1-3 random categories
# user_preferences = {
#     user: np.random.choice(range(num_categories), size=np.random.randint(1, 4), replace=False)
#     for user in range(num_users)
# }

# # Assign each item to a category
# item_categories = {item: np.random.choice(range(num_categories)) for item in range(num_items)}

# # Generate interactions
# user_ids = np.random.choice(range(num_users), num_interactions)
# item_ids = np.random.choice(range(num_items), num_interactions)

# # Generate ratings based on the pattern
# ratings = []
# for user, item in zip(user_ids, item_ids):
#     item_category = item_categories[item]
#     if item_category in user_preferences[user]:
#         ratings.append(np.random.choice([3, 4]))  # High rating for preferred categories
#     else:
#         ratings.append(np.random.choice([1, 2]))  # Low rating otherwise

# # Introduce noise
# num_noisy = int(noise_ratio * num_interactions)
# noisy_indices = np.random.choice(range(num_interactions), num_noisy, replace=False)
# for idx in noisy_indices:
#     ratings[idx] = np.random.choice(range(1, 6))  # Replace with random rating

# # Combine into a DataFrame
# data = pd.DataFrame({
#     'user_id': user_ids,
#     'product_id': item_ids,
#     'category': [item_categories[item] for item in item_ids],
#     'rating': ratings
# })
# print("Artificial Dataset Preview:")

########################### ver 3 ##############################

# Parameters
# num_users = 100 # Number of unique users
# num_items = 50    # Number of unique items
# num_categories = 5  # Number of unique categories
# num_interactions = 10000  # Number of user-item interactions
# noise_ratio = 0.01  # Percentage of noisy interactions

# # Step 1: Define deterministic user preferences
# user_preferences = {user: user % num_categories for user in range(num_users)}

# # Step 2: Assign items to categories in a cyclic pattern
# item_categories = {item: item % num_categories for item in range(num_items)}

# # Step 3: Generate deterministic interactions
# user_ids = np.arange(num_interactions) % num_users  # Cycle through users
# item_ids = np.arange(num_interactions) % num_items  # Cycle through items

# # Step 4: Generate ratings based on the pattern
# ratings = []
# for user, item in zip(user_ids, item_ids):
#     preferred_category = user_preferences[user]
#     item_category = item_categories[item]
#     if item_category == preferred_category:
#         ratings.append(5)  # High rating for preferred category
#     else:
#         ratings.append(1)  # Low rating otherwise

# # Step 5: Introduce noise
# num_noisy = int(noise_ratio * num_interactions)
# noisy_indices = np.random.choice(range(num_interactions), num_noisy, replace=False)
# for idx in noisy_indices:
#     ratings[idx] = np.random.choice(range(1, 6))  # Replace with random rating

# # Step 6: Create a DataFrame
# data = pd.DataFrame({
#     'user_id': user_ids,
#     'product_id': item_ids,
#     'category': [item_categories[item] for item in item_ids],
#     'rating': ratings
# })

########################################################################
print(data.head())
 
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
print(len(symbols))
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

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(args.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < args.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - args.hypervector_size))

        for k in range(args.message_size * 2):
            if tm.ta_action(1, i, k):
                if k < args.message_size:
                    l.append("c%d" % (k))
                else:
                    l.append("NOT c%d" % (k - args.message_size))

        print(" AND ".join(l))

# print(graphs_test.hypervectors)
# print(tm.hypervectors)
# print(graphs_test.edge_type_id)
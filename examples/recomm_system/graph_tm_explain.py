from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import argparse
import numpy as np
import prepare_dataset


def main(args):  
    np.random.seed(42)
    data = prepare_dataset.aug_amazon_products(noise_ratio = args.dataset_noise_ratio)
    x, y = prepare_dataset.construct_x_y(data)
    X_train, X_test, Y_train, Y_test = prepare_dataset.train_test_split(x,y)
    print("No of classes: ", len(np.unique(Y_train)))
    print("Y classes: ", np.unique(Y_train))
    
    users = data['user_id'].unique()
    print("Users: ",len(users))
    
    items = data['product_id'].unique()
    print("Items: ",len(items))
    
    categories = data['category'].unique()
    print("Categories: ",len(categories))
    
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

    for epoch in range(args.epochs):
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    
    # print_clause_explanations(tm, graphs_train)
        
    state = tm.get_state()
    clause_weights_flat = state[1]
    number_of_classes = int(state[2])
    number_of_clauses = int(state[3])
    weights = clause_weights_flat.reshape(number_of_classes, number_of_clauses)
    # weights = tm.get_state()[1].reshape(2, -1)
    # Get Clauses in symbols format and Messages in clause_indices format
    clause_literals = tm.get_clause_literals(graphs_train.hypervectors).astype(np.int32)
    num_symbols = len(graphs_train.symbol_id)

    # Create symbol_id to symbol_name dictionary for printing symbol names
    symbol_dict = dict((v, k) for k, v in graphs_train.symbol_id.items())

    threshold = 7
    for target_label_of_Y in np.unique(Y_train):
        print(f"Target label: {target_label_of_Y}, Number of positive clauses: {np.sum(weights[target_label_of_Y]>0)}")
        for clause in range(tm.number_of_clauses):
            if weights[target_label_of_Y, clause] > 0:
                for literal in range(num_symbols):
                    state = clause_literals[clause, literal]
                    neg_state = clause_literals[clause, literal + num_symbols]
                    if np.any(state > threshold) or np.any(neg_state > threshold):
                        print(f"Clause {clause} [{weights[target_label_of_Y, clause]:>4d}]", end=": ")
                        if state > threshold:
                            print(f"{clause_literals[clause, literal]}{symbol_dict[literal]}", end=" ")

                        if neg_state > threshold:
                            print(f"~{clause_literals[clause, literal + num_symbols]}{symbol_dict[literal]}", end=" ")

                        print("")

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
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
    parser.add_argument("--max-included-literals", default=23, type=int)
    parser.add_argument("--dataset_noise_ratio", default=0.01, type=float)
    parser.add_argument("--exp_id", default="", type=str)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    # train
    main(default_args())
    
    # run just explanation from saved model
    # print_clause_explanations()
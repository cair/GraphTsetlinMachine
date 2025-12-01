from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import argparse
import numpy as np
import prepare_dataset
import os
import matplotlib.pyplot as plt


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

    # create output folder for plots
    os.makedirs("plots", exist_ok=True)
    top_n = 15  # number of top symbols to show per class
    threshold = 7  # threshold for literal inclusion
    
    class_labels = np.unique(Y_train)
    num_classes = len(class_labels)

    # --- Determine optimal figure and subplot spacing ---
    # Calculate dynamic height based on number of classes and top_n to avoid too much whitespace
    # Adjusted multiplier for height
    row_height_per_feature = 0.25 # Smaller value to reduce vertical space per feature
    fig_height = max(3.5 * num_classes, num_classes * top_n * row_height_per_feature + 2) 

    fig, axes = plt.subplots(
        nrows=num_classes, 
        ncols=2, 
        figsize=(10, fig_height), # Slightly reduced width, dynamically calculated height
        sharey='row',             # Share the Y-axis (symbol labels) only within each row
        gridspec_kw={
            'wspace': 0.15, # Reduced horizontal space between subplots
            'hspace': 0.45, # Reduced vertical space between rows of subplots
            'left': 0.1,    # Adjust left margin
            'right': 0.95,  # Adjust right margin
            'top': 0.93,    # Adjust top margin (for suptitle)
            'bottom': 0.05  # Adjust bottom margin (for common x-labels)
        }
    )

    # Ensure 'axes' is a 2D array even for a single class (num_classes=1)
    if num_classes == 1:
        axes = np.array([axes])

    # --- Main Plotting Loop ---
    for i, target_label_of_Y in enumerate(class_labels):
        
        # 1. Aggregate scores for the current class (Aggregation Logic remains the same)
        state_scores = np.zeros(num_symbols, dtype=float)
        weight_scores = np.zeros(num_symbols, dtype=float)
        
        for clause in range(tm.number_of_clauses):
            w = weights[target_label_of_Y, clause]
            if w <= 0:
                continue
            for literal in range(num_symbols):
                state = clause_literals[clause, literal]
                neg_state = clause_literals[clause, literal + num_symbols]
                
                if state > threshold:
                    state_scores[literal] += state
                    weight_scores[literal] += w
                
                if neg_state > threshold:
                    state_scores[literal] -= neg_state
                    weight_scores[literal] -= w

        # 2. Select top symbols
        idx_sorted = np.argsort(-np.abs(state_scores))
        top_idx = idx_sorted[:top_n]
        
        labels = [symbol_dict[i] for i in top_idx]
        state_vals = state_scores[top_idx]
        weight_vals = weight_scores[top_idx] 

        bar_positions = range(len(state_vals))
        
        # 3. Plot State Scores (Left Subplot of the current row)
        ax1 = axes[i, 0]
        state_colors = ['tab:green' if v > 0 else '#843d3a' for v in state_vals]
        ax1.barh(bar_positions, state_vals[::-1], color=[c for c in state_colors[::-1]])
        
        # Set Y-axis labels (features)
        ax1.set_yticks(bar_positions)
        ax1.set_yticklabels(labels[::-1], fontsize=7) # Smaller font for labels
        ax1.set_ylabel(f'Class {target_label_of_Y}\n(Features)', rotation=90, labelpad=5, fontsize=9) # Reduced labelpad
        
        # Titles
        if i == 0: # Only top row gets the full title
            ax1.set_title('Aggregate Clause State', fontsize=10)
        else:
            # For other rows, a simpler label on the Y-axis is enough, or no title
            ax1.set_title('') # Clear title for non-top rows

        ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--') 
        
        # 4. Plot Weight Scores (Right Subplot of the current row)
        ax2 = axes[i, 1]
        weight_colors = ['tab:blue' if v > 0 else '#41729a' for v in weight_vals] 
        ax2.barh(bar_positions, weight_vals[::-1], color=[c for c in weight_colors[::-1]])
        
        # Remove redundant Y-axis labels and ticks from the right subplot
        ax2.tick_params(axis='y', left=False, labelleft=False) 
        
        # Titles
        if i == 0: # Only top row gets the full title
            ax2.set_title('Aggregate Clause Weight ($w$)', fontsize=10)
        else:
            ax2.set_title('') # Clear title for non-top rows

        ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--') 

        # Add X-axis labels only to the bottom row for clarity
        if i == num_classes - 1:
            ax1.set_xlabel('Score Value', fontsize=9)
            ax2.set_xlabel('Score Value', fontsize=9)
        else:
            # Remove X-axis labels from non-bottom rows
            ax1.tick_params(axis='x', labelbottom=False)
            ax2.tick_params(axis='x', labelbottom=False)

    # Add a main title for the entire figure
    plt.suptitle(f'Top {top_n} Feature Importance Comparison Across All Classes', fontsize=14)
    # plt.tight_layout() # We are using specific gridspec_kw margins instead

    out_path = f"plots/all_classes_top{top_n}_comparison_compact.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved combined comparison plot: {out_path}")

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=1.0, type=float)
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
    main(default_args())
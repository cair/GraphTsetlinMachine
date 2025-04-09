import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from keras.api.datasets import mnist
from matplotlib import colors
from skimage.util import view_as_windows
from tqdm import tqdm

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train = X_train[:1000]
# Y_train = Y_train[:1000]
# X_test = X_test[:1000]
# Y_test = Y_test[:1000]

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)


def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=2, type=int)
	parser.add_argument("--number-of-clauses", default=2500, type=int)
	parser.add_argument("--T", default=3125, type=int)
	parser.add_argument("--s", default=10.0, type=float)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--depth", default=1, type=int)
	parser.add_argument("--hypervector-size", default=128, type=int)
	parser.add_argument("--hypervector-bits", default=2, type=int)
	parser.add_argument("--message-size", default=256, type=int)
	parser.add_argument("--message-bits", default=2, type=int)
	parser.add_argument("--double-hashing", dest="double_hashing", default=True, action="store_true")
	parser.add_argument("--max-included-literals", default=32, type=int)

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args


args = default_args()

patch_size = 10
dim = 28 - patch_size + 1

number_of_nodes = dim * dim

symbols = []

# Column and row symbols
for i in range(dim):
	symbols.append("C:%d" % (i))
	symbols.append("R:%d" % (i))

# Patch pixel symbols
for i in range(patch_size * patch_size):
	symbols.append(i)

graphs_train = Graphs(
	X_train.shape[0],
	symbols=symbols,
	hypervector_size=args.hypervector_size,
	hypervector_bits=args.hypervector_bits,
	double_hashing=args.double_hashing,
)

for graph_id in range(X_train.shape[0]):
	graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

for graph_id in range(X_train.shape[0]):
	for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
		graphs_train.add_graph_node(graph_id, node_id, 0)

graphs_train.prepare_edge_configuration()

for graph_id in range(X_train.shape[0]):
	if graph_id % 1000 == 0:
		print(graph_id, X_train.shape[0])

	windows = view_as_windows(X_train[graph_id, :, :], (patch_size, patch_size))
	for q in range(windows.shape[0]):
		for r in range(windows.shape[1]):
			node_id = q * dim + r

			patch = windows[q, r].reshape(-1).astype(np.uint32)
			for k in patch.nonzero()[0]:
				graphs_train.add_graph_node_property(graph_id, node_id, k)

			graphs_train.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
			graphs_train.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

graphs_train.encode()

print("Training data produced")

graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
	graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
	for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
		graphs_test.add_graph_node(graph_id, node_id, 0)

graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
	if graph_id % 1000 == 0:
		print(graph_id, X_test.shape[0])

	windows = view_as_windows(X_test[graph_id, :, :], (10, 10))
	for q in range(windows.shape[0]):
		for r in range(windows.shape[1]):
			node_id = q * dim + r

			patch = windows[q, r].reshape(-1).astype(np.uint32)
			for k in patch.nonzero()[0]:
				graphs_test.add_graph_node_property(graph_id, node_id, k)

			graphs_test.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
			graphs_test.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

graphs_test.encode()

print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=args.number_of_state_bits,
	depth=args.depth,
	message_size=args.message_size,
	message_bits=args.message_bits,
	max_included_literals=args.max_included_literals,
	double_hashing=False,
	grid=(16 * 13, 1, 1),
	block=(128, 1, 1),
)

for i in range(args.epochs):
	start_training = time()
	tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
	stop_testing = time()

	result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

	print(
		"%d %.2f %.2f %.2f %.2f"
		% (
			i,
			result_train,
			result_test,
			stop_training - start_training,
			stop_testing - start_testing,
		)
	)


def scale(X, x_min, x_max):
	nom = (X - X.min()) * (x_max - x_min)
	denom = X.max() - X.min()
	denom = denom + (denom == 0)
	return x_min + nom / denom


def scale_image(img):
	if len(img.shape) == 3:
		for ch in range(3):
			img[..., ch] = scale(img[..., ch], 0, 1)
	else:
		img = scale(img, 0, 1)

	return img


weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)

# Get Literals insymbol format
clause_literals = tm.get_clause_literals(graphs_train.hypervectors)
num_symbols = len(graphs_train.symbol_id)


def create_graph_sample(X):
	graph_sample = Graphs(X.shape[0], init_with=graphs_train)
	for graph_id in range(X.shape[0]):
		graph_sample.set_number_of_graph_nodes(graph_id, number_of_nodes)

	graph_sample.prepare_node_configuration()

	for graph_id in range(X.shape[0]):
		for node_id in range(graph_sample.number_of_graph_nodes[graph_id]):
			graph_sample.add_graph_node(graph_id, node_id, 0)

	graph_sample.prepare_edge_configuration()

	for graph_id in range(X.shape[0]):
		if graph_id % 1000 == 0:
			print(graph_id, X.shape[0])

		windows = view_as_windows(X[graph_id, :, :], (10, 10))
		for q in range(windows.shape[0]):
			for r in range(windows.shape[1]):
				node_id = q * dim + r

				patch = windows[q, r].reshape(-1).astype(np.uint32)
				for k in patch.nonzero()[0]:
					graph_sample.add_graph_node_property(graph_id, node_id, k)

				graph_sample.add_graph_node_property(graph_id, node_id, "C:%d" % (q))
				graph_sample.add_graph_node_property(graph_id, node_id, "R:%d" % (r))

	graph_sample.encode()

	return graph_sample


# get output of each clause at each node
clause_outputs, class_sums = tm.transform_nodewise(create_graph_sample(X_test[:1]))

# Lets consider example 0 of graph_test
for e in [0]:
	# print class sums and prediction
	pred = np.argmax(class_sums[e])
	print(f"{Y_test[e]=}")
	print(f"{class_sums[e]=}")
	print(f"{pred=}")

	# Number of position symbols according to how the images are encoded.
	position_symbols = 38
	total_symbols = len(graphs_test.symbol_id)

	# clause_outputs -> (num_samples, num_clauses, num_nodes)
	co = clause_outputs[e]

	# Store images for positive literals, negated literals and pos-neg literals
	final_imgs = np.zeros((3, 28, 28))
	for c in tqdm(range(tm.number_of_clauses)):
		w = weights[pred, c]

		# Ignore Negative polarity clauses
		if w < 0:
			continue

		# Split the literals
		pos_literals = clause_literals[c, position_symbols:total_symbols]
		neg_literals = clause_literals[c, total_symbols + position_symbols : 2 * total_symbols]

		eff_literals = pos_literals - neg_literals

		# Reshape into patch dimensions
		pos_literals = pos_literals.reshape((10, 10))
		neg_literals = neg_literals.reshape((10, 10))
		eff_literals = eff_literals.reshape((10, 10))

		# For each patch position in the image
		for node_id in range(np.max(graphs_test.number_of_graph_nodes)):
			# Get x,y from node_id
			xpos, ypos = node_id // 19, node_id % 19

			# If clause is active, then weight and add it
			if co[c, node_id] == 1:
				final_imgs[0, xpos : xpos + 10, ypos : ypos + 10] += pos_literals * w
				final_imgs[1, xpos : xpos + 10, ypos : ypos + 10] += neg_literals * w
				final_imgs[2, xpos : xpos + 10, ypos : ypos + 10] += eff_literals * w

	# Matplotlib visualization shenanigans, so that pos-neg image colors are shown properly
	# rocket = color_palette("rocket", as_cmap=True)      # Needs seaborn
	rocket = plt.get_cmap("magma")
	fullcmap = colors.LinearSegmentedColormap.from_list("fullcmap", rocket(np.linspace(0, 1, 100)))
	cmap = colors.LinearSegmentedColormap.from_list("cmap", rocket(np.linspace(0.5, 1, 50)))

	fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="compressed", squeeze=False)
	axs[0, 0].imshow(X_test[e])
	axs[0, 1].imshow(final_imgs[0], cmap=cmap)
	axs[0, 2].imshow(final_imgs[1], cmap=cmap)
	axs[0, 3].imshow(final_imgs[2], cmap=fullcmap)

	axs[0, 0].set_title("Input image")
	axs[0, 1].set_title("Positive Literals")
	axs[0, 2].set_title("Negative Literals")
	axs[0, 3].set_title("Pos-Neg Literals")

	for ax in axs.ravel():
		ax.axis("off")

	plt.show()

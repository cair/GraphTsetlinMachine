# Tsetlin Machine for Logical Learning and Reasoning With Graphs

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2024?style=flat-square)

Implementation of the Graph Tsetlin Machine.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Tutorial](#tutorial)
  - [Initialization](#initialization)
  - [Adding the Nodes](#adding-the-nodes)
  - [Adding the Edges](#adding-the-edges)
  - [Adding the Properties and Class Labels](#adding-the-properties-and-class-labels)
- [Graph Tsetlin Machine Basics](#graph-tsetlin-machine-basics)
  - [Clause-Driven Message Passing](#clause-driven-message-passing)
  - [Logical Learning and Reasoning With Nested Clauses](#logical-learning-and-reasoning-with-nested-clauses)
- [Demos](#demos)
  - [Vanilla MNIST](#Vanilla-MNIST)
  - [Convolutional MNIST](#Convolutional-MNIST)
- [Paper](#paper)
- [CUDA Configurations](#cuda-configurations)
- [Roadmap](#roadmap)
- [Licence](#licence)

## Features

- Processes directed and labeled [multigraphs](https://en.wikipedia.org/wiki/Multigraph)
- [Vector symbolic](https://link.springer.com/article/10.1007/s10462-021-10110-3) node properties and edge types
- Nested (deep) clauses
- Arbitrarily sized inputs
- Incorporates [Vanilla](https://tsetlinmachine.org/wp-content/uploads/2022/11/Tsetlin_Machine_Book_Chapter_One_Revised.pdf), Multiclass, [Convolutional](https://tsetlinmachine.org/wp-content/uploads/2023/12/Tsetlin_Machine_Book_Chapter_4_Convolution.pdf), and [Coalesced](https://arxiv.org/abs/2108.07594) [Tsetlin Machines](https://tsetlinmachine.org)
- Rewritten faster CUDA kernels 

## Installation

```bash
pip3 install graphtsetlinmachine
```
or
```bash
python ./setup.py sdist
pip3 install dist/GraphTsetlinMachine-0.2.6.tar.gz
```

## Tutorial 

In this tutorial, you create graphs for the Noisy XOR problem and then train and test the Graph Tsetlin Machine on these.

Noisy XOR gives four kinds of graphs, shown below:

<p align="center">
  <img width="60%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/NoisyXOR.png">
</p>

Observe how each node in a graph has one of two properties: **A** or **B**. If both of the graph's nodes have the same property, the graph is given the class label _Y=0_. Otherwise, it is given the class label _Y=1_.

The task of the Graph Tsetlin Machine is to assign the correct class label to each graph when the labels used for training are noisy.

### Initialization

Start by creating the training graphs using the _Graphs_ construct:
```bash
graphs_train = Graphs(
    10000,
    symbols = ['A', 'B'],
    hypervector_size = 32,
    hypervector_bits = 2
)
```
You initialize the graphs as follows:
- *Number of Graphs.* The first number sets how many graphs you are going to create. Here, you prepare for creating _10,000_ graphs.

- *Symbols.* Next, you find the symbols **A** and **B**. You use these symbols to assign properties to the nodes of the graphs. You can define as many symbols as you like. For the Noisy XOR problem, you only need two.

- *Vector Symbolic Representation (Hypervectors).* You also decide how large hypervectors you would like to use to store the symbols. Larger hypervectors room more symbols. Since you only have two symbols, set the size to _32_. Finally, you decide how many bits to use for representing each symbol. Use _2_ bits for this tutorial. You then get _32*31/2 = 496_ unique bit pairs - plenty of space for two symbols!
  
- *Generation and Compilation.* The generation and compilation of hypervectors happen automatically during initialization, using [sparse distributed codes](https://ieeexplore.ieee.org/document/917565).

### Adding the Nodes

The next step is to set how many nodes you want in each of the _10,000_ graphs you are building. For the Noisy XOR problem, each graph has two nodes:
```bash
for graph_id in range(10000):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)
```
After doing that, you prepare for adding the nodes:
```bash
graphs_train.prepare_node_configuration()
```
You add the two nodes to the graphs as follows, giving them one outgoing edge each:
```bash
for graph_id in range(10000):
  number_of_outgoing_edges = 1

  graphs_train.add_graph_node(graph_id, 'Node 1', number_of_outgoing_edges)

  graphs_train.add_graph_node(graph_id, 'Node 2', number_of_outgoing_edges)
```

### Adding the Edges

You are now ready to prepare for adding edges:
```bash
graphs_train.prepare_edge_configuration()
```

After that, you connect the two nodes of each graph with two edges:
```bash
for graph_id in range(10000):
    edge_type = "Plain"
    graphs_train.add_graph_node_edge(graph_id, 'Node 1', 'Node 2', edge_type)
    graphs_train.add_graph_node_edge(graph_id, 'Node 2', 'Node 1', edge_type)
```
You need two edges because you build directed graphs, and with two edges you cover both directions. We use only one type of edges for this, which we name _Plain_.

### Adding the Properties and Class Labels

In the last step, you randomly assign property *A* or *B* to each node.
```bash
Y_train = np.empty(10000, dtype=np.uint32)
for graph_id in range(10000):
    x1 = random.choice(['A', 'B'])
    x2 = random.choice(['A', 'B'])

    graphs_train.add_graph_node_property(graph_id, 'Node 1', x1)
    graphs_train.add_graph_node_property(graph_id, 'Node 2', x2)
```
Based on this assignment, you set the class label of the graph. If both nodes get the same property, the class label is _0_. Otherwise, it is _1_.
```bash
    if x1 == x2:
        Y_train[graph_id] = 0
    else:
        Y_train[graph_id] = 1
```
The class label is finally randomly inverted to introduce noise.
```bash
    if np.random.rand() <= 0.01:
        Y_train[graph_id] = 1 - Y_train[graph_id]
```
## Graph Tsetlin Machine Basics

### Clause-Driven Message Passing

The Graph Tsetlin Machine is based on message passing. As illustrated below, a pool of clauses examines each node in the graph. Whenever a clause matches the properties of a node, it sends a message about its finding through the node's outgoing edges.

<p align="center">
  <img width="75%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/MessagePassing.png">
</p>
When a node receives a message, it adds the message to its properties. In this manner, the messages supplement the node properties with contextual information.

### Logical Learning and Reasoning With Nested Clauses

The above message passing enables logical learning and reasoning with nested (deep) clauses. The number of message rounds decides the depth of the reasoning. Three layers of reasoning, for instance, consist of local reasoning, followed by two rounds of message passing, illustrated below:

<p align="center">
  <img width="100%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/DeepLogicalLearningAndReasoning.png">
</p>

Initially, the clauses only consider the nodes' properties (marked in black).
* In the first round of message passing, matching clauses send out their messages. These messages supplement the receiving node's properties (marked in red).
* In the second round, the clauses examine the nodes again, now taking into account the first round of messages. Based on this revisit, the clauses produce the second round of messages, marked in blue.
  
This process continues until reaching the desired depth of reasoning, in this case depth three. Finally, the Tsetlin Automata Teams update their states based on how the clauses handled the classification task.

Notice how each team operates across a node's properties as well as the incorporated messages.  In this manner, they are able to build nested clauses. That is, a clause can draw upon the outcomes of other clauses to create hierarchical clause structures, centered around the various nodes. Hence, the power of the scheme!

## Demos

### Vanilla MNIST

The Graph Tsetlin Machine supports rich data (images, video, text, spectrograms, sound, etc.). One can, for example, add an entire image to a node, illustrated for MNIST images below:

<p align="center">
  <img width="40%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/VanillaMNIST.png">
</p>

Here, each white pixel in the grid of <i>28x28</i> pixels gets its own symbol: W<sub>x,y</sub>. You define an image by adding its white pixels as properties to the graph node. Note that with only a single node, you obtain a Coalesced Vanilla Tsetlin Machine. See the Vanilla MNIST Demo in the example folder for further details.

### Convolutional MNIST

By using many nodes to capture rich data, you can exploit inherent structure in the data. Below, each MNIST image is broken down into a grid of _19x19_ image patches, each patch containing _10x10_ pixels:

<p align="center">
  <img width="60%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/ConvolutionalMNIST.png">
</p>

Again, white pixel symbols W<sub>x,y</sub> define the image content. However, this example also shows how you can use a node's location inside the image to enhance the representation. You do this by introducing row R<sub>x</sub> and column C<sub>y</sub> symbols. These symbols allow the Graph Tsetlin Machine to learn and reason about pixel patterns as well as their location inside the image. Without adding any edges, the result is a Coalesced Convolutional Tsetlin Machine. See the Convolutional MNIST Demo in the example folder for further details.

## Paper

_A Tsetlin Machine for Logical Learning and Reasoning With Graphs_. Ole-Christoffer Granmo, et al., 2024. (Forthcoming)

## CUDA Configurations

### DGX-2 and A100

```bash
tm = MultiClassGraphTsetlinMachine(
  ...
  grid=(16*13,1,1),
  block=(128,1,1)
)
```

### DGX H100

```bash
tm = MultiClassGraphTsetlinMachine(
  ...
  grid=(16*13*4,1,1),
  block=(128,1,1)
)
```

## Roadmap

- Rewrite graphs.py in C or numba for much faster construction of graphs
- Add autoencoder
- Add regression
- Add multi-output
- Graph initialization with adjacency matrix

## Licence

Copyright (c) 2024 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

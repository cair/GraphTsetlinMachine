# Tsetlin Machine for Logical Learning and Reasoning With Graphs (Work in Progress)

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2024?style=flat-square)

Implementation of the Graph Tsetlin Machine.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Tutorial](#tutorial)
  - [Initialization](#initialization)
  - [Clause-Driven Message Passing](#messagepassing)
  - [Learning and Reasoning With Nested Clauses](#nestedclauses)
- [Demos](#demos)
- [Paper](#paper)
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
pip3 install dist/GraphTsetlinMachine-0.2.5.tar.gz
```

## Tutorial 

In this tutorial, you create graphs for the Noisy XOR problem and then train and test the Graph Tsetlin Machine on these. You have four kinds of graphs, shown below:

<p align="center">
  <img width="50%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/NoisyXOR.png">
</p>

Each node has one of two properties: _'A'_ or _'B'_. If both of the nodes in a graph have the same property, the graph is labeled _Y=0_. Otherwise, it is labeled _Y=1_. The task of the Graph Tsetlin Machine is to assign the correct label to each type of graph, when the labels used for training are noisy.

### Initialization

Start by creating the training graphs using the _Graphs_ class:
```bash
graphs_train = Graphs(
    10000,
    symbols = ['A', 'B'],
    hypervector_size = 32,
    hypervector_bits = 2
)
```
You initialize the class as follows:
- *Number of Graphs.* The first number sets how many graphs you are going to create. Here, you prepare for creating _10,000_ graphs.

- *Symbols.* Next, you find the symbols 'A' and 'B'. You use these symbols to assign properties to the nodes of the graphs. You can define as many symbols as you like. For the XOR problem, you only need two.

- *Vector Symbolic Representation (Hypervectors).* You also decide how large hypervectors you would like to use to store the symbols. Larger hypervectors room more symbols. Since you only have two symbols, set the size to _32_. Finally, you decide how many bits to use for representing each symbol. Use _2_ bits for this tutorial. You then get _32*31/2 = 496_ unique bit pairs - plenty of space for two symbols!
  
- *Generation and Compilation.* The generation and compilation of hypervectors happen automatically during initialization of your _Graphs_ object,  using [sparse distributed codes](https://ieeexplore.ieee.org/document/917565).

### Adding the Nodes

The next step is to set how many nodes you want in each of the _10,000_ graphs you are building. For the NoisyXOR problem, each graph has two nodes:
```bash
for graph_id in range(10000):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)
```
After doing that, you prepare for adding the nodes:
```bash
graphs_train.prepare_node_configuration()
```
The two nodes of each graph are automatically named _0_ and _1_, respectively. You add them to the graphs as follows, giving them one outgoing edge each:
```bash
for graph_id in range(10000):
  number_of_outgoing_edges = 1

  node_id = 0
  graphs_train.add_graph_node(graph_id, node_id, number_of_outgoing_edges)

  node_id = 1
  graphs_train.add_graph_node(graph_id, node_id, number_of_outgoing_edges)
```

### Clause-Driven Message Passing

<p align="center">
  <img width="75%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/MessagePassing.png">
</p>

### Learning and Reasoning With Nested Clauses

<p align="center">
  <img width="100%" src="https://github.com/cair/GraphTsetlinMachine/blob/master/figures/DeepLogicalLearningAndReasoning.png">
</p>

## Demos

Demos coming soon.

## Paper

_A Tsetlin Machine for Logical Learning and Reasoning With Graphs_. Ole-Christoffer Granmo, et al., 2024. (Forthcoming)

## Roadmap

- Rewrite graphs.py in C or numba for much faster construction of graphs
- Add Tsetlin Machine Autoencoder
- Add Tsetlin Machine Regression

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

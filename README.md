# Tsetlin Machine for Logical Learning and Reasoning With Graphs (Work in Progress)

![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Python Version](https://img.shields.io/pypi/pyversions/interpret.svg?style=flat-square)![Maintenance](https://img.shields.io/maintenance/yes/2024?style=flat-square)

Implementation of the Graph Tsetlin Machine.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Tutorial](#tutorial)
  - [Clause-Driven Message Passing](#)
  - [Learning and Reasoning With Nested Clauses](#nestedclauses)
- [Demos](#demos)
- [Licence](#licence)

## Features

- Processes labeled directed [multigraphs](https://en.wikipedia.org/wiki/Multigraph)
- [Vector symbolic](https://link.springer.com/article/10.1007/s10462-021-10110-3) node properties and edge types using [sparse distributed codes](https://ieeexplore.ieee.org/document/917565)
- Nested (deep) clauses
- Arbitrarily sized inputs
- Incorporates [Vanilla](https://tsetlinmachine.org/wp-content/uploads/2022/11/Tsetlin_Machine_Book_Chapter_One_Revised.pdf), Multiclass, [Convolutional](https://tsetlinmachine.org/wp-content/uploads/2023/12/Tsetlin_Machine_Book_Chapter_4_Convolution.pdf), and [Coalesced](https://arxiv.org/abs/2108.07594) Tsetlin Machine (regression and auto-encoding supported soon)
- Rewritten faster CUDA kernels 

## Installation

```bash
python ./setup.py sdist
pip3 install dist/GraphTsetlinMachine-0.2.4.tar.gz
```

## Tutorial 

In this tutorial, we will create graphs for the Noisy XOR problem and then train the Graph Tsetlin Machine on these graphs. Start by creating the training graphs using the _Graphs_ class:
```bash
graphs_train = Graphs(
    10000,
    symbol_names = ['A', 'B'],
    hypervector_size = 32,
    hypervector_bits = 2
)
```

### Symbols

The first number is how many graphs you are going to create. Here, we will create 10 000 graphs. Next, you find the symbols 'A' and 'B'. You use these symbols to assign properties to the nodes of each graph. You can define as many symbols as you like. For the XOR problem, we only need two.

### Vector Symbolic Representation (Hypervectors)

You also decide how large hypervectors you would like to use to store the symbols. Larger hypervectors room more symbols. Since you only have two symbols, set the size to 32. Finally, you decide how many bits to use for representing each symbol. You can use 2 bits here. The hypervector can then represent up to _32*31/2 = 496_ unique symbols. The generation and compilation of hypervectors happen automatically during initialization of your _Graphs_ object. 

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

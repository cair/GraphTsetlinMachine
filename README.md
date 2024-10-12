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

- Processes multigraphs
- [Vector symbolic](https://link.springer.com/article/10.1007/s10462-021-10110-3) node properties and edge types
- Nested (deep) clauses
- Arbitrarily sized inputs
- Incorporates Vanilla, Multiclass, Convolutional, and Coalesced Tsetlin Machine (Regression and auto-encoding supported soon)
- Rewritten faster CUDA kernels 

## Installation

```bash
python ./setup.py sdist
pip3 install dist/GraphTsetlinMachine-0.2.4.tar.gz
```

## Tutorial 

In this tutorial, we will create graphs for the Noisy XOR problem and then train the Graph Tsetlin Machine on these graphs. Start by creating the training graphs:
```bash
graphs_train = Graphs(
    10000,
    symbol_names=['A', 'B'],
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)
```
The first number is how many graphs you are going to create. Here, we will create 10 000 graphs. Next, you find the symbols 'A' and 'B'. You use these symbols to assign properties to the nodes of a graph. You can define as many symbols as you like. We here only need two to capture the XOR problem.

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

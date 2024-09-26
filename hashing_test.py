import argparse
import numpy as np
from sympy import prevprime

parser = argparse.ArgumentParser()
parser.add_argument("--items", default=3000, type=int)
parser.add_argument("--inserts", default=100, type=int)
parser.add_argument("--bits", default=16384, type=int)

args = parser.parse_args()

prime = prevprime(args.bits // 2)

collision = {}
collisions = 0
for i in range(args.items):
    first_bit = i % (args.bits // 2)
    second_bit = prime - (i % prime)

    if (first_bit, second_bit) in collision:
        collisions += 1
    collision[(first_bit, second_bit)] = 1
#    collision[(second_bit, first_bit)] = 1

collision_local = {}
collisions_insert = 0
for j in range(args.inserts):
    i = np.random.randint(args.items)
    first_bit = i % (args.bits // 2)
    second_bit = prime - (i % prime)

    if (first_bit in collision_local) and (second_bit in collision_local):
        collisions_insert += 1

    collision_local[first_bit] = 1
    collision_local[second_bit] = 1

print(collisions_insert, collisions)

collision = {}
collisions = 0
for i in range(args.items):
    sqrt_i = int(np.sqrt(i))
    sqrt_i_2 = sqrt_i**2

    if i - sqrt_i_2 < sqrt_i:
        first_bit = i - sqrt_i_2
        second_bit = sqrt_i
    else:
        second_bit = i - sqrt_i_2 - sqrt_i
        first_bit = sqrt_i

    first_bit = first_bit % args.bits
    second_bit = second_bit % args.bits

    if (first_bit, second_bit) in collision:
        collisions += 1
    collision[(first_bit, second_bit)] = 1
    collision[(second_bit, first_bit)] = 1

collision_first = {}
collision_second = {}
collisions_insert = 0
for j in range(args.inserts):
    i = np.random.randint(args.items)
 
    sqrt_i = int(np.sqrt(i))
    sqrt_i_2 = sqrt_i**2

    if i - sqrt_i_2 < sqrt_i:
        first_bit = i - sqrt_i_2
        second_bit = sqrt_i
    else:
        second_bit = i - sqrt_i_2 - sqrt_i
        first_bit = sqrt_i

    first_bit = first_bit % args.bits
    second_bit = second_bit % args.bits

    if (first_bit in collision_local) and (second_bit in collision_local):
        collisions_insert += 1

    collision_local[first_bit] = 1
    collision_local[second_bit] = 1

print(collisions_insert, collisions)

hypervectors = {}
collisions = 0
indexes = np.arange(args.bits)
for i in range(args.items):
    (first_bit, second_bit) = np.random.choice(indexes, size=(2), replace=False)

    if (first_bit, second_bit) in collision:
        collisions += 1

    hypervectors[(first_bit, second_bit)] = 1

    collision[(first_bit, second_bit)] = 1
    collision[(second_bit, first_bit)] = 1

collision_local = {}
collisions_insert = 0

hypervectors = list(hypervectors.keys())
for j in range(args.inserts):
    i = np.random.randint(len(hypervectors))

    (first_bit, second_bit) = hypervectors[i]

    if (first_bit in collision_local) and (second_bit in collision_local):
        collisions_insert += 1

    collision_local[first_bit] = 1
    collision_local[second_bit] = 1

print(collisions_insert, collisions)

hypervectors = {}
collision = {}
collisions = 0
indexes = np.arange(args.bits)
for i in range(args.items):
    (first_bit, second_bit, third_bit) = np.random.choice(indexes, size=(3), replace=False)

    if (first_bit, second_bit, third_bit) in collision:
        collisions += 1

    hypervectors[(first_bit, second_bit, third_bit)] = 1

    collision[(first_bit, second_bit, third_bit)] = 1
    collision[(second_bit, first_bit, third_bit)] = 1
    collision[(third_bit, second_bit, first_bit)] = 1
    collision[(first_bit, third_bit, second_bit)] = 1
    collision[(second_bit, third_bit, first_bit)] = 1
    collision[(third_bit, first_bit, second_bit)] = 1

collision_local = {}
collisions_insert = 0

hypervectors = list(hypervectors.keys())
for j in range(args.inserts):
    i = np.random.randint(len(hypervectors))

    (first_bit, second_bit, third_bit) = hypervectors[i]

    if (first_bit in collision_local) and (second_bit in collision_local) and (third_bit in collision_local):
        collisions_insert += 1

    collision_local[first_bit] = 1
    collision_local[second_bit] = 1
    collision_local[third_bit] = 1

print(collisions_insert, collisions)
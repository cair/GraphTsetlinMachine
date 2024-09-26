import argparse
import numpy as np
from sympy import prevprime
import mmh3

def murmur(key, seed):
    for i in range(4):
        seed ^= key[i]
        seed *= 0x5bd1e995
        seed ^= seed >> 15
    return seed

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

collisions_insert_average = 0
for k in range(1000):

    collision_first = {}
    collision_second = {}

    collisions_insert = 0
    for j in range(args.inserts):
        i = np.random.randint(args.items)
        first_bit = i % (args.bits // 2)
        second_bit = prime - (i % prime)

        if (first_bit in collision_first) and (second_bit in collision_second):
            collisions_insert += 1

        collision_first[first_bit] = 1
        collision_second[second_bit] = 1

    collisions_insert_average += collisions_insert

print(collisions_insert_average / 1000, collisions)

collision = {}
collisions = 0
for i in range(args.items):
    first_bit = i % (args.bits // 2)
    second_bit = prime - (i % prime)
    third_bit = (i // 11) % (args.bits // 2)

    if (first_bit, second_bit, third_bit) in collision:
        collisions += 1
    collision[(first_bit, second_bit, third_bit)] = 1

collisions_insert_average = 0
for k in range(1000):

    collision_first = {}
    collision_second = {}
    collision_third = {}

    collisions_insert = 0
    for j in range(args.inserts):
        i = np.random.randint(args.items)
        first_bit = i % (args.bits // 2)
        second_bit = prime - (i % prime)
        third_bit = (i // 11) % (args.bits // 2)

        if (first_bit in collision_first) and (second_bit in collision_second) and (third_bit in collision_third):
            collisions_insert += 1

        collision_first[first_bit] = 1
        collision_second[second_bit] = 1
        collision_third[third_bit] = 1

    collisions_insert_average += collisions_insert

print(collisions_insert_average / 1000, collisions)

collision = {}
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

hypervectors = list(hypervectors.keys())
collisions_insert_average = 0
for k in range(1000):
    collision_local = {}
    collisions_insert = 0

    for j in range(args.inserts):
        i = np.random.randint(len(hypervectors))

        (first_bit, second_bit) = hypervectors[i]

        if (first_bit in collision_local) and (second_bit in collision_local):
            collisions_insert += 1

        collision_local[first_bit] = 1
        collision_local[second_bit] = 1
    collisions_insert_average += collisions_insert/1000.0

print(collisions_insert_average, collisions)

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

hypervectors = list(hypervectors.keys())
collisions_insert_average = 0
for k in range(1000):
    collision_local = {}
    collisions_insert = 0

    for j in range(args.inserts):
        i = np.random.randint(len(hypervectors))

        (first_bit, second_bit, third_bit) = hypervectors[i]

        if (first_bit in collision_local) and (second_bit in collision_local) and (third_bit in collision_local):
            collisions_insert += 1

        collision_local[first_bit] = 1
        collision_local[second_bit] = 1
        collision_local[third_bit] = 1
    
    collisions_insert_average += collisions_insert/1000.0

print(collisions_insert_average, collisions)

collision = {}
collisions = 0
for i in range(args.items):
    i = np.uint32(i).tobytes()
    first_bit = murmur(i, 0x81726354) % (args.bits)
    second_bit = murmur(i, 0x12345678) % (args.bits)
    third_bit = murmur(i, 0x87654321) % (args.bits)

    if (first_bit, second_bit, third_bit) in collision:
        collisions += 1

    collision[(first_bit, second_bit, third_bit)] = 1
    collision[(second_bit, first_bit, third_bit)] = 1
    collision[(third_bit, second_bit, first_bit)] = 1
    collision[(first_bit, third_bit, second_bit)] = 1
    collision[(second_bit, third_bit, first_bit)] = 1
    collision[(third_bit, first_bit, second_bit)] = 1

collisions_insert_average = 0
for k in range(1000):
    collision_local = {}
    collisions_insert = 0

    for j in range(args.inserts):
        i = np.random.randint(args.items, dtype=np.uint32).tobytes()

        first_bit = murmur(i, 0x81726354) % (args.bits)
        second_bit = murmur(i, 0x12345678) % (args.bits)
        third_bit = murmur(i, 0x87654321) % (args.bits)

        if (first_bit in collision_local) and (second_bit in collision_local) and (third_bit in collision_local):
            collisions_insert += 1

        collision_local[first_bit] = 1
        collision_local[second_bit] = 1
        collision_local[third_bit] = 1
    
    collisions_insert_average += collisions_insert/1000.0

print(collisions_insert_average, collisions)
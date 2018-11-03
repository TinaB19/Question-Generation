from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def parse_args(description = "QG"):
    import argparse
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("-embedding", type = str, required = True)
    parser.add_argument("-dict", type = str, required = True)
    parser.add_argument("-output", type = str, required = True)
    parser.add_argument("-seed", type = int, default = 19941023)
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


def main():
    args = parse_args()

    word2embedding = {}
    dimension = None
    with open(args.embedding, "r") as input_file:
        for line in input_file:
            line = line.split()
            word2embedding[line[0]] = np.asarray(map(float, line[1 : ]))
            dimension = len(line) - 1

    with open(args.dict, "r") as input_file:
        words = [ line.split()[0] for line in input_file ]

    embedding = np.random.uniform(low = -1.0 / 3, high = 1.0 / 3, size = (len(words), dimension))
    embedding = np.asarray(embedding, dtype = np.float32)
    unknown_count = 0
    for i, word in enumerate(words):
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word, ))
            unknown_count += 1

    t = torch.from_numpy(embedding)
    torch.save(t, args.output)
    print("Total unknown: %d" % (unknown_count, ))


if __name__ == "__main__":
    main()

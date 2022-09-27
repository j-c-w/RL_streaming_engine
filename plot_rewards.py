import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(fn):
    with open(fn) as f:
        vs = []
        for line in f.readlines():
            v = float(line.split(' ')[-1].strip())
            if v < -0.005:
                #There are some 0s when the thing doesn't find an initial palcement
                vs.append(v)

    window=100
    if len(vs) > 200:
        vs = np.convolve(vs, np.ones(window), 'valid') / float(window)
    plt.plot(range(0, len(vs)), vs, label=fn)
    plt.legend()
    # plt.yscale('log')
    plt.savefig('output.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')

    args = parser.parse_args()
    for file in args.files:
        plot(file)

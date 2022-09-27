import json
import matplotlib.pyplot as plt
import argparse

def load_line(l):
    d = json.loads(l)
    return d

def load_lines(f):
    data = []
    with open(f) as f:
        for line in f.readlines():
            data.append(load_line(line))
    return data

def plot_line(data):
    xvs = range(len(data))
    data = [max(d['episode_reward_mean'], -65) for d in data]

    plt.plot(xvs, data)
    plt.savefig('rewards.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", nargs='+')
    args = parser.parse_args()

    for j in args.result_json:
        data = load_lines(j)
        plot_line(data)

import matplotlib.pyplt as plt
import numpy as np
import argparse

class RunData():
    def __init__(self, numbers, rewards):
        self.numbers = numbers
        self.rewards = rewards

# Draw a CDF of the performance of each different
# architecture configuration tried for each CGRA.
def draw_performance_cdf(names, data):
    for d in data:

# Parse the output files.
def load_data(fname):
    cgra_numbers = []
    total_rewards = []

    with open(fname, 'r') as f:
        for line in f.readlines():
            if 'Looking at cgra' in line:
                cgra_no = line.split(' ')[5]
                cgra_numbers.append(int(cgra_no))

            if 'Total reward was' in line:
                reward = line.split(' ')[6]
                total_rewards.append(float(reward))

    data = RunData(cgra_numbers, total_rewards)
    return fname, data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('argfiles', nargs='+')

    args = parser.parse_args()

    names = []
    datas = []
    for f in args.argfiles:
        n, d = load_data(f)

        names.append(n)
        datas.append(d)

    draw_performance_cdf(names, datas)
import matplotlib.pyplot as plt
import argparse
import numpy as np

def load_data(rfile):
    data = []
    with open(rfile) as f:
        for line in f.readlines():
            try:
                data.append(int(line.split(' ')[-1]))
            except:
                pass
    return data

def plot(data, dataml):
    xvs = sorted(data)
    print("(Normal) Min is {min}, num runs is {runs}".format(min=xvs[0], runs=len(xvs)))
    yvs = np.linspace(0.0, 1.0, len(xvs))

    mlxvs = sorted(dataml)
    mlyvs = np.linspace(0.0, 1.0, len(mlxvs))
    print("(RL) Min is {min}, num runs is {runs}".format(min=mlxvs[0], runs=len(mlxvs)))

    plt.plot(xvs, yvs, label='Simulated Annealing')
    plt.plot(mlxvs, mlyvs, label='ML-Assisted')
    plt.legend()
    plt.xlim(min(xvs), max(xvs))
    plt.ylim(0, 1)
    plt.savefig('annealing_results.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_annealer_rfile')
    parser.add_argument('ml_file')
    args = parser.parse_args()

    data = load_data(args.sim_annealer_rfile)
    dataml = load_data(args.ml_file)
    plot(data, dataml)

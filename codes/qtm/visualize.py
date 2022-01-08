from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
def plot_15layer(title, fidelitiesghz, fidelitiesw, fidelitieshaar, is_save = False):
    x = [1,2,3,4,5]
    plt.plot(x, fidelitiesghz, marker='o', linestyle='--', color='blue', label = 'GHZ')
    plt.plot(x, fidelitiesw, marker='o', linestyle='--', color='orange', label = 'W')
    plt.plot(x, fidelitieshaar, marker='o', linestyle='--', color='green', label = 'Haar')
    plt.title(title)
    plt.xlabel('Number of layers')
    plt.xticks(np.arange(min(x) - 1, max(x) + 5, 1))
    plt.yticks(np.arange(0, 1.4, 0.2))
    plt.ylabel('Value')
    plt.legend()
    if is_save:
        plt.savefig('num_layer_' + title +'.png', format='png', dpi=600)
    plt.show()

def read_15layer(path, dirs):
    fidelitiesghz = []
    fidelitiesw = []
    fidelitieshaar = []
    tracesghz = []
    tracesw = []
    traceshaar = []
    for dir in dirs:
        dir = str(dir)
        fidelities_ghz = pd.read_csv(path + dir + "/fidelities_ghz.csv", sep=",", header=None)
        fidelities_ghz = fidelities_ghz.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitiesghz.append(np.real(np.max(fidelities_ghz)))

        fidelities_w = pd.read_csv(path + dir + "/fidelities_w.csv", sep=",", header=None)
        fidelities_w = fidelities_w.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitiesw.append(np.real(np.max(fidelities_w)))

        fidelities_haar = pd.read_csv(path + dir + "/fidelities_haar.csv", sep=",", header=None)
        fidelities_haar = fidelities_haar.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitieshaar.append(np.real(np.max(fidelities_haar)))

        traces_ghz = pd.read_csv(path + dir + "/traces_ghz.csv", sep=",", header=None).values
        tracesghz.append(np.min(traces_ghz))
        traces_w = pd.read_csv(path + dir + "/traces_w.csv", sep=",", header=None).values
        tracesw.append(np.min(traces_w))
        traces_haar = pd.read_csv(path + dir + "/traces_haar.csv", sep=",", header=None).values
        traceshaar.append(np.min(traces_haar))
    return fidelitiesghz, fidelitiesw, fidelitieshaar, tracesghz, tracesw, traceshaar

def read_15layer_last(path, dirs):
    fidelitiesghz = []
    fidelitiesw = []
    fidelitieshaar = []
    tracesghz = []
    tracesw = []
    traceshaar = []
    for dir in dirs:
        dir = str(dir)
        fidelities_ghz = pd.read_csv(path + dir + "/fidelities_ghz.csv", sep=",", header=None)
        fidelities_ghz = fidelities_ghz.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitiesghz.append(np.real(fidelities_ghz[-1]))

        fidelities_w = pd.read_csv(path + dir + "/fidelities_w.csv", sep=",", header=None)
        fidelities_w = fidelities_w.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitiesw.append(np.real(fidelities_w[-1]))

        fidelities_haar = pd.read_csv(path + dir + "/fidelities_haar.csv", sep=",", header=None)
        fidelities_haar = fidelities_haar.applymap(lambda s: complex(s.replace('i', 'j'))).values
        fidelitieshaar.append(np.real(fidelities_haar[-1]))

        traces_ghz = pd.read_csv(path + dir + "/traces_ghz.csv", sep=",", header=None).values
        tracesghz.append(traces_ghz[-1])
        traces_w = pd.read_csv(path + dir + "/traces_w.csv", sep=",", header=None).values
        tracesw.append(traces_w[-1])
        traces_haar = pd.read_csv(path + dir + "/traces_haar.csv", sep=",", header=None).values
        traceshaar.append(traces_haar[-1])
    return fidelitiesghz, fidelitiesw, fidelitieshaar, tracesghz, tracesw, traceshaar
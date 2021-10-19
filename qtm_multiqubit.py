from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
from matplotlib import pyplot as plt



def create_ghz_state(qc, theta):
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

def u_thetas_multiqubit(qc, thetas):
    for i in range(0, qc.num_qubits):
        qc.rz(thetas[i][0], i)
        qc.rx(thetas[i][1], i)
        qc.rz(thetas[i][2], i)
    return qc

def entanglement_multiqubit(qc):
    for i in range(0, qc.num_qubits): 
        if i == qc.num_qubits - 1:
            qc.cnot(qc.num_qubits - 1, 0)
        else:
            qc.cnot(i, i + 1)
    return qc


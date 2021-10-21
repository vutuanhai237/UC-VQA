from qiskit import *
import numpy as np
from matplotlib import pyplot as plt
from qtm import measure
import qiskit.quantum_info as qi

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

def u_cluster_3qubit(qc, thetas):
    qc = u_thetas_multiqubit(qc, thetas[0])
    qc = entanglement_multiqubit(qc)
    qc = u_thetas_multiqubit(qc, thetas[1])
    return qc

def u_cluster(qc, n_layer, thetas):
    for i in range(0, n_layer):
        qc = entanglement_multiqubit(qc)
        qc = u_thetas_multiqubit(qc, thetas[i])
    return qc

def grad_u_cluster_multiqubit(qc, n_layer, thetas, r, s, counter="0"):
    gradient_l = np.zeros((thetas).shape)
    for i in range(0, thetas.shape[0]):
        for j in range(0, thetas.shape[1]):
            for k in range(0, thetas.shape[2]):
                thetas1, thetas2 = thetas.copy(), thetas.copy()
                thetas1[i, j, k] += s
                thetas2[i, j, k] -= s
                qc1 = u_cluster(qc.copy(), n_layer, thetas1)
                qc2 = u_cluster(qc.copy(), n_layer, thetas2)
                gradient_l[i, j, k] = -r*(
                    measure(qc1, range(qc1.num_qubits), range(qc1.num_qubits), counter) - 
                    measure(qc2, range(qc2.num_qubits), range(qc2.num_qubits), counter))
    return gradient_l




def grad_l_multiqubit(qc, create_qc_func, thetas, r, s, counter="0"):
    gradient_l = np.zeros((thetas).shape)
    for i in range(0, thetas.shape[0]):
        for j in range(0, thetas.shape[1]):
            for k in range(0, thetas.shape[2]):
                thetas1, thetas2 = thetas.copy(), thetas.copy()
                thetas1[i, j, k] += s
                thetas2[i, j, k] -= s
                qc1 = create_qc_func(qc.copy(), thetas1)
                qc2 = create_qc_func(qc.copy(), thetas2)
                gradient_l[i, j, k] = -r*(
                    measure(qc1, range(qc1.num_qubits), range(qc1.num_qubits), counter) - 
                    measure(qc2, range(qc2.num_qubits), range(qc2.num_qubits), counter))
    return gradient_l

def get_psi_hat_3qubit(thetas):
    qc = QuantumCircuit(3, 3)
    qc = u_cluster_3qubit(qc, thetas).inverse()
    return qi.Statevector.from_instruction(qc)


def get_psi_hat_multiqubit(n_qubit, n_layer, thetas):
    qc = QuantumCircuit(n_qubit, n_qubit)
    qc = u_cluster(qc, n_layer, thetas).inverse()
    return qi.Statevector.from_instruction(qc)
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

def u_cluster_4qubit(qc, thetas):
    qc = entanglement_multiqubit(qc)
    qc = u_thetas_multiqubit(qc, thetas[0])
    return qc




def grad_l_multiqubit(qc, create_qc_func, thetas, r, s):
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
                    measure(qc1, range(qc.num_qubits), range(qc.num_qubits), "000") - 
                    measure(qc2, range(qc.num_qubits), range(qc.num_qubits), "000"))
    return gradient_l

def get_psi_hat_3qubit(thetas):
    qc = QuantumCircuit(3, 3)
    qc = u_cluster_3qubit(qc, thetas).inverse()
    return qi.Statevector.from_instruction(qc)
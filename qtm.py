from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
from qiskit.visualization import plot_bloch_multivector, plot_bloch_vector
from scipy.linalg import sqrtm
import constant

def u_thetas(qc, thetas, qubit = 0):
    qc.rz(thetas[0], qubit)
    qc.rx(thetas[1], qubit)
    qc.rz(thetas[2], qubit)
    return qc

def u_thetas_h(qc, thetas, qubit = 0):
    qc.rz(thetas[0], qubit)
    qc.rx(thetas[1], qubit)
    qc.rz(thetas[2], qubit)
    qc.h(qubit)
    return qc



def get_psi_hat(thetas):
    qc = QuantumCircuit(1, 1)
    qc = u_thetas(qc, thetas).inverse()
    return qi.Statevector.from_instruction(qc)

def get_psi_hat_x_basis(thetas):
    qc = QuantumCircuit(1,1)
    qc = u_thetas_h(qc, thetas).inverse()
    return qi.Statevector.from_instruction(qc)


def u_3(qc, theta, phi, lambdaz, index):
    qc.u3(theta, phi, lambdaz, index)
    return qc

def measure(qc, qubits, cbits, counter):
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], cbits[i])
    qobj = assemble(qc, shots = constant.shots)  
    counts = (Aer.get_backend('qasm_simulator')).run(qobj).result().get_counts()
    return counts.get(counter, 0) / constant.shots



def grad_l(qc, thetas, r, s, measurement_basis = 'z'):
    gradient_l = np.zeros(len(thetas))
    for i in range(0, len(thetas)):
        thetas1, thetas2 = thetas.copy(), thetas.copy()
        thetas1[i] += s
        thetas2[i] -= s
        if measurement_basis == 'z':
            qc1 = u_thetas(qc.copy(), thetas1, 0)
            qc2 = u_thetas(qc.copy(), thetas2, 0)
        if measurement_basis == 'x':
            qc1 = u_thetas_h(qc.copy(), thetas1, 0)
            qc2 = u_thetas_h(qc.copy(), thetas2, 0)
        gradient_l[i] = -r*(measure(qc1, [0], [0], "0") - measure(qc2, [0], [0], "0"))
    return gradient_l


def trace_distance(rho_psi, rho_psi_hat):
    """Since density matrices are Hermitian,
    so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of 
    (rho_psi - rho_psi_hat) matrix

    Args:
        rho_psi (DensityMatrix): psi,
        rho_psi_hat (DensityMatrix): psi hat
    """
    w, v = np.linalg.eig((rho_psi - rho_psi_hat).data)
    return (1/2*sum(abs(w)))

def trace_fidelity(rho_psi, rho_psi_hat):
    rho_psi = rho_psi.data
    rho_psi_hat = rho_psi_hat.data
    return np.trace(sqrtm((sqrtm(rho_psi)).dot(rho_psi_hat)).dot(sqrtm(rho_psi)))

def inner_product(psi_hat, psi):
    return ((psi_hat.conjugate()).transpose()).dot(psi)

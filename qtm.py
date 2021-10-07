from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
from qiskit.visualization import plot_bloch_multivector, plot_bloch_vector
from scipy.linalg import sqrtm
import constant

def u_thetas(qc, thetas, index = 0):
    qc.rz(thetas[0], 0)
    qc.rx(thetas[1], 0)
    qc.rz(thetas[2], 0)
    return qc
    
def get_psi_hat(thetas):
    # a = np.exp(1j/2*(thetas[0]+thetas[2]))*np.cos(thetas[1]/2)
    # b = 1j*np.exp(1j/2*(thetas[0]-thetas[2]))*np.sin(thetas[1]/2)
    # return np.array([a, b])
    qc = QuantumCircuit(1, 1)
    qc = u_thetas(qc, thetas).inverse()
    return qi.Statevector.from_instruction(qc)

def u_3(qc, theta, phi, lambdaz, index):
    qc.u3(theta, phi, lambdaz, index)
    return qc
def construct_circuit(qc, thetas, index = 0):
    qc = QuantumCircuit(1, 1)
    qc = u_thetas(thetas, qc)
    return qc

def measure(qc, i):
    """Get P0 values by measurement
    Args:
        qc (QuantumCircuit)
    Returns:
        float: P0
    """
    qobj = assemble(qc, shots = constant.shots)  
    counts = (Aer.get_backend('qasm_simulator')).run(qobj).result().get_counts()
    return counts['0'] / constant.shots

def grad_l(qc, thetas):
    gradient_l = np.zeros(len(thetas))
    for i in range(0, len(thetas)):
        thetas1, thetas2 = thetas.copy(), thetas.copy()
        thetas1[i] += np.pi/2
        thetas2[i] -= np.pi/2
      
        qc1 = u_thetas(qc.copy(), thetas1, 0)
        qc1.measure(0, 0)

        qc2 = u_thetas(qc.copy(), thetas2, 0)
        qc2.measure(0, 0)

        gradient_l[i] = -1/2*(measure(qc1, 1) - measure(qc2, 1))

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

def inner_product(psi, psi_hat):
    return ((psi_hat.conjugate()).transpose()).dot(psi)


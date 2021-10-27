import qiskit
import numpy as np
import qtm.base_qtm, qtm.qtm_1qubit

def create_ghz_state(qc: qiskit.QuantumCircuit, theta: float):
    """Create GHZ state with a parameter

    Args:
        qc (QuantumCircuit): Init circuit
        theta (Float): Parameter
    
    Returns:
        QuantumCircuit
    """
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

def u_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Create enhanced version of qtm.qtm_1qubit.u_1q

    Args:
        qc (QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
    
    Returns:
        QuantumCircuit
    """
    j = 0
    for i in range(0, qc.num_qubits):
        qc = qtm.qtm_1qubit.u_1q(qc, thetas[j:j + 3], i)
        j = j + 3
    return qc

def entangle_nqubit(qc: qiskit.QuantumCircuit):
    """Create entanglement state

    Args:
        qc (QuantumCircuit)
    
    Returns:
        QuantumCircuit
    """
    for i in range(0, qc.num_qubits): 
        if i == qc.num_qubits - 1:
            qc.cnot(qc.num_qubits - 1, 0)
        else:
            qc.cnot(i, i + 1)
    return qc

def u_cluster_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Create a complicated u gate multi-qubit

    Args:
        qc (QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters

    Returns:
        QuantumCircuit
    """
    qc = u_nqubit(qc, thetas[0:qc.num_qubits * 3])
    qc = entangle_nqubit(qc)
    qc = u_nqubit(qc, thetas[qc.num_qubits * 3:])
    return qc

def u_cluster(qc, n_layer, thetas):
    for i in range(0, n_layer):
        qc = entanglement_multiqubit(qc)
        qc = u_thetas_multiqubit(qc, thetas[i])
    return qc

def grad_u_cluster_multiqubit(qc, n_layer, thetas, r, s):
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
                    qtm.base_qtm.measure(qc1, range(qc1.num_qubits), range(qc1.num_qubits), base_qtm.get_counter(qc1.num_qubits)) - 
                    qtm.base_qtm.measure(qc2, range(qc2.num_qubits), range(qc2.num_qubits), base_qtm.get_counter(qc2.num_qubits)))
    return gradient_l


def get_u_cluster_nqubit_hat(thetas, num_qubits: int = 1):
    """Get psi_hat of u_nqubit

    Args:
        thetas (Numpy array): Parameters
        num_qubits (int, optional): Number of qubits. Defaults to 1.

    Returns:
        Statevector: State vectpr of u_nqubit_dagger
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = u_cluster_nqubit(qc, thetas).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)


def get_psi_hat_multiqubit(n_qubit, n_layer, thetas):
    qc = qiskit.QuantumCircuit(n_qubit, n_qubit)
    qc = u_cluster(qc, n_layer, thetas).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)
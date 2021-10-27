import qiskit
import qtm.base_qtm, qtm.qtm_1qubit

def create_ghz_state(qc: qiskit.QuantumCircuit, theta: float):
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter
    
    Returns:
        - QuantumCircuit: the added circuit
    """
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

def u_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Create enhanced version of qtm.qtm_1qubit.u_1q

    Args:
        - qc (QuantumCircuit): Init circuit
        - thetas (Numpy array): Parameters
    
    Returns:
        - QuantumCircuit: the added circuit
    """
    j = 0
    for i in range(0, qc.num_qubits):
        qc = qtm.qtm_1qubit.u_1q(qc, thetas[j:j + 3], i)
        j = j + 3
    return qc

def entangle_nqubit(qc: qiskit.QuantumCircuit):
    """Create entanglement state

    Args:
        - qc (QuantumCircuit): Init circuit
    
    Returns:
        - QuantumCircuit: the added circuit
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
        - qc (QuantumCircuit): Init circuit
        - thetas (Numpy array): Parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    qc = u_nqubit(qc, thetas[0:qc.num_qubits * 3])
    qc = entangle_nqubit(qc)
    qc = u_nqubit(qc, thetas[qc.num_qubits * 3:])
    return qc

def u_cluster_nlayer_nqubit(qc: qiskit.QuantumCircuit, thetas, num_layers):
    """Create a complicated u gate multi-qubit

    Args:
        - qc (QuantumCircuit): Init circuit
        - thetas (Numpy array): Parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    params_per_layer = int(len(thetas) / num_layers)
    for i in range(0, num_layers):
        qc = entangle_nqubit(qc)
        qc = u_nqubit(qc, thetas[i * params_per_layer:(i + 1) * params_per_layer])
    return qc

def get_u_cluster_nqubit_hat(thetas, num_qubits: int = 1):
    """Get psi_hat of u_nqubit

    Args:
        - thetas (Numpy array): Parameters
        - num_qubits (int, optional): Number of qubits. Defaults to 1.

    Returns:
        - Statevector: State vectpr of u_nqubit_dagger
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = u_cluster_nqubit(qc, thetas).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)


def get_u_cluster_nlayer_nqubit_hat(thetas, num_qubits: int = 1, num_layers: int = 1):
    """Get psi_hat of u_nlayer_nqubit

    Args:
        - thetas (Numpy array): Parameters
        - num_qubits (int, optional): Number of qubits. Defaults to 1.

    Returns:
        - Statevector: State vectpr of u_nqubit_dagger
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = u_cluster_nlayer_nqubit(qc, thetas, num_layers).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)
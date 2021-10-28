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
        qc = qtm.qtm_1qubit.u_1qubit(qc, thetas[j:j + 3], i)
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

def create_rx_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Add a R_X layer

    Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Mumpu array): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits):
        qc.rx(thetas[i], i)
    return qc

def create_rz_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Add a R_Z layer

    Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Mumpu array): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits):
        qc.rz(thetas[i], i)
    return qc

def create_cry_nqubit(qc: qiskit.QuantumCircuit, thetas):
    """Create control Control-RY state

     Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Mumpu array): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1, 2): 
        qc.cry(thetas[i], i, i + 1)
    for i in range(1, qc.num_qubits - 1, 2): 
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[qc.num_qubits - 1], qc.num_qubits - 1, 0)        
    return qc
def create_cry_nqubit_inverse(qc: qiskit.QuantumCircuit, thetas):
    """Create control Control-RY state but swap control and target qubit

     Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Mumpu array): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    for i in range(1, qc.num_qubits - 1, 2): 
        qc.cry(thetas[i], i + 1, i)
    qc.cry(thetas[qc.num_qubits - 1], 0, qc.num_qubits - 1)        
    return qc

def create_arbitrary_nqubit(qc: qiskit.QuantumCircuit, thetas, num_layers: int = 1):
    """An arbitrary layer for initial quantum state

    Args:
        qc (qiskit.QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if len(thetas) != num_layers * n * 5:
        raise Exception('Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i:(i + 1)*n*5]
        qc = create_rx_nqubit(qc, phis[:n])
        qc = create_cry_nqubit_inverse(qc, phis[n:n*2])
        qc = create_rz_nqubit(qc, phis[n*2:n*3])
        qc = create_cry_nqubit(qc, phis[n*3:n*4])
        qc = create_rz_nqubit(qc, phis[n*4:n*5])
    return qc

def create_arbitrary_nqubit2(qc, thetas):
    qc.rx(thetas[0], 0)
    qc.ry(thetas[1], 0)
    qc.rx(thetas[2], 1)
    qc.ry(thetas[3], 1)
    qc.rx(thetas[4], 2)
    qc.ry(thetas[5], 2)
    qc.rxx(thetas[6], 1, 2)
    qc.rzx(thetas[7], 1, 2)
    qc.rxx(thetas[8], 0, 1)
    qc.rzx(thetas[9], 0, 1)
    qc.rxx(thetas[10], 1, 2)
    qc.rzx(thetas[11], 1, 2)
    qc.rxx(thetas[12], 0, 1)
    qc.rzx(thetas[13], 0, 1)
    return qc

import qiskit
import qtm.base_qtm, qtm.qtm_1qubit
import numpy as np



###########################
######## GHZ State ########
###########################

def create_ghz_state(qc: qiskit.QuantumCircuit, theta: float = np.pi/2):
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter
    
    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

def create_ghz_state_inverse(qc: qiskit.QuantumCircuit, theta: float = np.pi/2):
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter
    
    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, qc.num_qubits - i - 1)
    qc.ry(-theta, 0)
    return qc

def create_GHZchecker_koczor(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
    """Create circuit includes koczor and GHZ

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): params
        - num_layers (int)
        - theta (float): param for general GHZ

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(theta, float) != True:
        theta = theta['theta']
    # |psi_gen> = U_gen|000...> 
    qc = create_koczor_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is GHZ state
    qc = create_ghz_state_inverse(qc, theta)
    return qc

def create_GHZchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
    """Create circuit includes binho and GHZ

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): params
        - num_layers (int)
        - theta (float): param for general GHZ

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(theta, float) != True:
        theta = theta['theta']
    # |psi_gen> = U_gen|000...> 
    qc = create_binho_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is GHZ state
    qc = create_ghz_state_inverse(qc, theta)
    return qc

###########################
######### W State #########
###########################

def create_w_state_3qubit_inverse(qc: qiskit.QuantumCircuit, theta: float = np.pi/2):
    """Create W inverse state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter
    
    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc.x(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    qc.ch(0, 1)
    qc.ry(-theta, 0) 
    return qc

def create_Wchecker_koczor(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
    """Create circuit includes koczor and W

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): params
        - num_layers (int)
        - theta (float): param for general W

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(theta, float) != True:
        theta = theta['theta']
    # |psi_gen> = U_gen|000...> 
    qc = create_koczor_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is W state
    qc = create_w_state_3qubit_inverse(qc, theta)
    return qc


def create_Wchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
    """Create circuit includes binho and W

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): params
        - num_layers (int)
        - theta (float): param for general W

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(theta, float) != True:
        theta = theta['theta']
    # |psi_gen> = U_gen|000...> 
    qc = create_binho_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is W state
    qc = create_w_state_3qubit_inverse(qc, theta)
    return qc

###########################
######## Haar State #######
###########################

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

def create_koczor_state(qc: qiskit.QuantumCircuit, thetas, num_layers: int = 1):
    """Create koczor anzsats 

    Args:
        qc (qiskit.QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
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


def create_haarchecker_koczor(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, encoder):
    """Create circuit includes haar and koczor

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): params
        - num_layers (int): num_layer for koczor
        - encoder: encoder for haar

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(encoder, qtm.encoding.Encoding) != True:
        encoder = encoder['encoder']
    qc1 = qiskit.QuantumCircuit(encoder.quantum_data)
    qc1 = create_koczor_state(qc1, thetas, num_layers = num_layers)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1

###########################
######## Binho State ######
###########################

def create_wy(qc: qiskit.QuantumCircuit, thetas):
    """Create WY state

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

def create_binho_state(qc: qiskit.QuantumCircuit, thetas, num_layers: int = 1):
    """Create koczor anzsats 

    Args:
        qc (qiskit.QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * n * 5:
        raise Exception('Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i:(i + 1)*n*5]
        qc = create_rx_nqubit(qc, phis[:n])
        qc = create_wy(qc, phis[n:n*2])
        qc = create_rz_nqubit(qc, phis[n*2:n*3])
        qc = create_wy(qc, phis[n*3:n*4])
        qc = create_rz_nqubit(qc, phis[n*4:n*5])
    return qc





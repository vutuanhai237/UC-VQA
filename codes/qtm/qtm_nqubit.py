import qiskit
import qtm.base_qtm, qtm.qtm_1qubit, qtm.custom_gate
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

def create_GHZchecker_alternating_layered(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
    """Create circuit includes alternating layered and GHZ

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
    qc = create_alternating_layerd_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is GHZ state
    qc = create_ghz_state_inverse(qc, theta)
    return qc
    
###########################
######### W State #########
###########################

def w(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0):
    """The below codes is implemented from [this paper](https://arxiv.org/abs/1606.09290)
    \n Simplest case: 3 qubits. <img src='../images/general_w.png' width = 500px/>
    \n General case: more qubits. <img src='../images/general_w2.png' width = 500px/>

    Args:
        - qc (qiskit.QuantumCircuit): [description]
        - num_qubits (int): [description]
        - shift (int, optional): [description]. Defaults to 0.

    Raises:
        - ValueError: When the number of qubits is not valid

    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1/np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cnot(i + shift, shift)
    return qc

def create_w_state(qc: qiskit.QuantumCircuit):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc.barrier()
    qc.x(0)
    
    qc = w(qc, qc.num_qubits)
    return qc

def create_w_state_inverse(qc: qiskit.QuantumCircuit):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc1.x(0)
    qc1 = w(qc1, qc.num_qubits)
    qc = qc.combine(qc1.inverse())
    return qc

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


def create_Wchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
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

    # |psi_gen> = U_gen|000...> 
    qc = create_binho_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is W state
    qc = create_w_state_inverse(qc)
    return qc


def create_Wchecker_alternating_layered(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    """Create circuit includes Alternating layered and W

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

    # |psi_gen> = U_gen|000...> 
    qc = create_alternating_layerd_state(qc, thetas, num_layers = num_layers)
    # U_target^t|psi_gen> with U_target is W state
    qc = create_w_state_inverse(qc)
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
    """Create koczor ansatz 

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


def create_haarchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, encoder):
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
    qc1 = create_binho_state(qc1, thetas, num_layers = num_layers)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1


def create_haarchecker_alternating_layered(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, encoder):
    """Create circuit includes Alternating layered and koczor

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
    qc1 = create_alternating_layerd_state(qc1, thetas, num_layers = num_layers)
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
    """Create koczor ansatz 

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

###########################
###### Layered State ######
###########################

def create_ry_nqubit(qc: qiskit.QuantumCircuit, thetas, shift = 0):
    """Add a R_Y layer

    Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Numpy array): parameters
        - shift (Int): start index
    Returns:
        - qiskit.QuantumCircuit
    """
    if qc.num_qubits - shift < len(thetas):
        raise Exception('Number of parameters must be equal num_qubits - shift')
    # for i in range(0, 0 + shift):
    #     qc.i(i)
    for i in range(0, len(thetas)):
        qc.ry(thetas[i], i + shift)
    # for i in range(shift + len(thetas), qc.num_qubits):
    #     qc.i(i)
    return qc

def create_swap_nqubit(qc: qiskit.QuantumCircuit, shift = 0):
    """Add a SWAP layer

    Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Numpy array): parameters
        - shift (Int): start index
    Returns:
        - qiskit.QuantumCircuit
    """
    # for i in range(0, 0 + shift):
    #     qc.i(i)
    for i in range(0 + shift, qc.num_qubits - 1, 2):
        qc.swap(i, i + 1)
    # if (qc.num_qubits - shift) % 2 == 1:
    #     qc.i(qc.num_qubits - 1)
    return qc

def create_alternating_layerd_state(qc: qiskit.QuantumCircuit, thetas, num_layers: int = 1):
    """Create Alternating layerd ansatz

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
    if len(thetas) != num_layers * (n * 5 - 4):
        raise Exception('Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i*(n*5 - 4):(i + 1)*(n*5 - 4)]
        qc.barrier()
        qc = create_ry_nqubit(qc, phis[:n])
        qc = create_swap_nqubit(qc)
        qc = create_ry_nqubit(qc, phis[n:n*2 - 1])
        qc = create_swap_nqubit(qc, shift = 1)
        qc = create_ry_nqubit(qc, phis[n*2 - 1:n*3 - 2], shift = 1)
        qc = create_swap_nqubit(qc)
        qc = create_ry_nqubit(qc, phis[n*3 - 2:n*4 - 3])
        qc = create_swap_nqubit(qc, shift = 1)
        qc = create_ry_nqubit(qc, phis[n*4 - 3:n*5 - 4], shift = 1)
    return qc


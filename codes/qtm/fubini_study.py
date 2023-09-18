import qiskit
import numpy as np
import qtm.constant
import qtm.ansatz
import qtm.utilities
import typing
import types
import scipy


def create_observers(qc: qiskit.QuantumCircuit, k: int = 0):
    """Return dictionary of observers

    Args:
        - qc (qiskit.QuantumCircuit): Current circuit
        - k (int, optional): Number of observers each layer. Defaults to qc.num_qubits.

    Returns:
        - Dict
    """
    if k == 0:
        k = qc.num_qubits
    observer = []
    for gate in (qc.data)[-k:]:
        gate_name = gate[0].name
        # Non-param gates
        if gate_name in ['barrier', 'swap']:
            continue
        # 2-qubit param gates
        if gate[0].name in ['crx', 'cry', 'crz', 'cx', 'cz']:
            # Take controlled wire as index
            wire = qc.num_qubits - 1 - gate[1][1].index
            # Take control wire as index
            # wire = qc.num_qubits - 1 - gate[1][0].index
        # Single qubit param gates
        else:
            wire = qc.num_qubits - 1 - gate[1][0].index
        observer.append([gate_name, wire])
    return observer


def calculate_g(qc: qiskit.QuantumCircuit, observers: typing.Dict[str, int]):
    """Fubini-Study tensor. Detail informations: 
    \n https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html

    Args:
        - qc (qiskit.QuantumCircuit): Current quantum circuit
        - observers (Dict[str, int]): List of observer type and its acting wire

    Returns:
        - numpy array: block-diagonal submatrix g
    """
    # Get |psi>
    psi = qiskit.quantum_info.Statevector.from_instruction(qc).data
    psi = np.expand_dims(psi, 1)
    # Get <psi|
    psi_hat = np.transpose(np.conjugate(psi))
    num_observers = len(observers)
    num_qubits = qc.num_qubits
    g = np.zeros([num_observers, num_observers], dtype=np.complex128)
    # Each K[j] must have 2^n x 2^n dimensional with n is the number of qubits
    Ks = []
    # Observer shorts from high to low
    for observer_name, observer_wire in observers:
        observer = qtm.constant.generator[observer_name]
        if observer_wire == 0:
            K = observer
        else:
            if observer_name in ['crx', 'cry', 'crz', 'cz']:
                K = qtm.constant.generator['11']
            else:
                K = qtm.constant.generator['i']
        for i in range(1, num_qubits):
            if i == observer_wire:
                K = np.kron(K, observer)
            else:
                if observer_name in ['crx', 'cry', 'crz', 'cz']:
                    K = np.kron(K, qtm.constant.generator['11'])
                else:
                    K = np.kron(K, qtm.constant.generator['i'])
        Ks.append(K)
    for i in range(0, num_observers):
        for j in range(0, num_observers):
            g[i, j] = psi_hat @ (Ks[i] @ Ks[j]) @ psi - (
                psi_hat @ Ks[i] @ psi) * (psi_hat @ Ks[j] @ psi)
            if g[i, j] < 10**(-10):
                g[i, j] = 0
    return g


######################################
## General quantum natural gradient ##
######################################

def get_wires_of_gate(gate: typing.Tuple):
    """Get index bit that gate act on

    Args:
        - gate (qiskit.QuantumGate): Quantum gate

    Returns:
        - numpy arrray: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register.index)
    return list_wire


def is_gate_in_list_wires(gate: typing.Tuple, wires: typing.List):
    """Check if a gate lies on the next layer or not

    Args:
        - gate (qiskit.QuantumGate): Quantum gate
        - wires (numpy arrray): list of index bits

    Returns:
        - Bool
    """
    list_wire = get_wires_of_gate(gate)
    for wire in list_wire:
        if wire in wires:
            return True
    return False


def split_into_layers(qc: qiskit.QuantumCircuit):
    """Split a quantum circuit into layers

    Args:
        - qc (qiskit.QuantumCircuit): origin circuit

    Returns:
        - list: list of list of quantum gates
    """
    layers = []
    layer = []
    wires = []
    is_param_layer = None
    for gate in qc.data:
        name = gate[0].name
        if name in qtm.constant.ignore_generator:
            continue
        param = gate[0].params
        wire = get_wires_of_gate(gate)
        if is_param_layer is None:
            if len(param) == 0:
                is_param_layer = False
            else:
                is_param_layer = True
        # New layer's condition: depth increase or convert from non-parameterized layer to parameterized layer or vice versa
        if is_gate_in_list_wires(gate, wires) or (is_param_layer == False and len(param) != 0) or (is_param_layer == True and len(param) == 0):
            if is_param_layer == False:
                # First field is 'Is parameterized layer or not?'
                layers.append((False, layer))
            else:
                layers.append((True, layer))
            layer = []
            wires = []
        # Update sub-layer status
        if len(param) == 0 or name == 'state_preparation_dg':
            is_param_layer = False
        else:
            is_param_layer = True
        for w in wire:
            wires.append(w)
        layer.append((name, param, wire))
    # Last sub-layer
    if is_param_layer == False:
        # First field is 'Is parameterized layer or not?'
        layers.append((False, layer))
    else:
        layers.append((True, layer))
    return layers


def add_layer_into_circuit(qc: qiskit.QuantumCircuit, layer: typing.List):
    """Based on information in layers, adding new gates into current circuit

    Args:
        - qc (qiskit.QuantumCircuit): calculating circuit
        - layer (list): list of gate's informations

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    for name, param, wire in layer:
        if name == 'rx':
            qc.rx(param[0], wire[0])
        if name == 'ry':
            qc.ry(param[0], wire[0])
        if name == 'rz':
            qc.rz(param[0], wire[0])
        if name == 'crx':
            qc.crx(param[0], wire[0], wire[1])
        if name == 'cry':
            qc.cry(param[0], wire[0], wire[1])
        if name == 'crz':
            qc.crz(param[0], wire[0], wire[1])
        if name == 'cz':
            qc.cz(wire[0], wire[1])
    return qc


def qng_hessian(uvdagger: qiskit.QuantumCircuit, thetas: np.ndarray):
    alpha = 0.01
    n = uvdagger.num_qubits
    length = thetas.shape[0]
    thetas_origin = thetas

    def f(thetas):
        qc = uvdagger.bind_parameters(thetas)
        qc_reverse = uvdagger.bind_parameters(thetas_origin).inverse()
        qc = qc.compose(qc_reverse)
        return qtm.base.measure(qc, list(range(qc.num_qubits)))
    G = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(0, length):
        for j in range(0, length):
            k1 = f(thetas + alpha*(qtm.utilities.unit_vector(i,
                   length) + qtm.utilities.unit_vector(j, length)))
            k2 = -f(thetas + alpha * (qtm.utilities.unit_vector(i,
                    length) - qtm.utilities.unit_vector(j, length)))
            k3 = -f(thetas - alpha * (qtm.utilities.unit_vector(i,
                    length) - qtm.utilities.unit_vector(j, length)))
            k4 = f(thetas - alpha*(qtm.utilities.unit_vector(i,
                   length) + qtm.utilities.unit_vector(j, length)))
            G[i][j] = (1/(4*(np.sin(alpha))**2))*(k1 + k2 + k3 + k4)
    return -1/2*np.asarray(G)


def qng(uvaddger: qiskit.QuantumCircuit):
    """Calculate G matrix in qng

    Args:
        - qc (qiskit.QuantumCircuit)
        - thetas (np.ndarray): parameters
        - create_circuit_func (FunctionType)
        - num_layers (int): number of layer of ansatz

    Returns:
        - np.ndarray: G matrix
    """
    n = uvaddger.num_qubits
    # List of g matrices
    gs = []
    # Splitting circuit into list of V and W sub-layer (non-parameter and parameter)
    layers = split_into_layers(uvaddger)
    qc = qiskit.QuantumCircuit(n, n)
    for is_param_layer, layer in layers:
        if is_param_layer:
            observers = qtm.fubini_study.create_observers(
                add_layer_into_circuit(qc.copy(), layer), len(layer))
            gs.append(qtm.fubini_study.calculate_g(qc, observers))
        # Add next sub-layer into the current circuit
        qc = add_layer_into_circuit(qc, layer)

    G = gs[0]
    for i in range(1, len(gs)):
        G = scipy.linalg.block_diag(G, gs[i])
    return G

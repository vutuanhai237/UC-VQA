from ast import FunctionType
import qiskit
import numpy as np
from qiskit.visualization.text import Ex
import qtm.constant
import qtm.nqubit
from typing import Dict, Tuple, List
from scipy.linalg import block_diag


def is_duplicate_wire(v, gate_wire):
    for gate_name, gate_param[0], wire in v:
        if gate_wire == wire:
            return True
    return False


def find_observers(qc: qiskit.QuantumCircuit):
    vs = []
    v = []
    ws = []
    w = []
    i = 0
    for gate in qc.data:

        gate_name = gate[0].name
        gate_param = gate[0].params
        # Non-param gates
        if gate_name == 'barrier':
            continue
        # 2-qubit param gates
        if gate[0].name in ['crx', 'cry', 'crz', 'cx']:
            wire = qc.num_qubits - 1 - gate[1][1].index
        # Single qubit param gates
        else:
            wire = qc.num_qubits - 1 - gate[1][0].index
        print(wire)
        if is_duplicate_wire(v, wire):
            print('....')
            vs.append(v)
            v = []
        v.append([gate_name, gate_param[0], wire])


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
        if gate[0].name in ['crx', 'cry', 'crz', 'cx']:
            # Take controlled wire as index
            # wire = qc.num_qubits - 1 - gate[1][1].index
            # Take control wire as index
            wire = qc.num_qubits - 1 - gate[1][0].index
        # Single qubit param gates
        else:
            wire = qc.num_qubits - 1 - gate[1][0].index
        observer.append([gate_name, wire])
    return observer


def calculate_g(qc: qiskit.QuantumCircuit, observers: Dict[str, int]):
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
            K = qtm.constant.generator['i']
        for i in range(1, num_qubits):
            if i == observer_wire:
                K = np.kron(K, observer)
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


def calculate_u3z_state(qc: qiskit.QuantumCircuit, thetas):
    """Create u3z ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if len(thetas) != 3:
        raise Exception('Number of parameters must be equal 3')
    gs = []
    # Sub-Layer 1
    qc_copy = qc.copy()
    qc_copy.rz(thetas[0], 0)
    observers = (create_observers(qc_copy))
    gs.append(calculate_g(qc, observers))
    qc.rz(thetas[0], 0)
    # Sub-Layer 2
    qc_copy = qc.copy()
    qc_copy.rx(thetas[1], 0)
    observers = (create_observers(qc_copy))
    gs.append(calculate_g(qc, observers))
    qc.rx(thetas[1], 0)
    # Sub-Layer 3
    qc_copy = qc.copy()
    qc_copy.rz(thetas[2], 0)
    observers = (create_observers(qc_copy))
    gs.append(calculate_g(qc, observers))
    qc.rz(thetas[2], 0)

    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_koczor_state(qc: qiskit.QuantumCircuit,
                           thetas,
                           num_layers: int = 1):
    """Create koczor ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * n * 5:
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 5')
    gs = []
    index_layer = 0
    for i in range(0, num_layers):
        phis = thetas[i * n * 5:(i + 1) * n * 5]
        # Sub-Layer 1
        qc_copy = qtm.nqubit.create_rx_nqubit(qc.copy(), phis[:n])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(qc, phis[:n])
        index_layer += 1
        # Sub-Layer 2
        qc_copy = qtm.nqubit.create_cry_nqubit_inverse(qc.copy(),
                                                       phis[n:n * 2])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_cry_nqubit_inverse(qc, phis[n:n * 2])
        index_layer += 1
        # Sub-Layer 3
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 2:n * 3])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 2:n * 3])
        index_layer += 1
        # Sub-Layer 4
        qc_copy = qtm.nqubit.create_cry_nqubit(qc.copy(), phis[n * 3:n * 4])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_cry_nqubit(qc, phis[n * 3:n * 4])
        index_layer += 1
        # Sub-Layer 5
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 4:n * 5])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 4:n * 5])
        index_layer += 1
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_binho_state(qc: qiskit.QuantumCircuit,
                          thetas,
                          num_layers: int = 1):
    """Create binho ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * n * 5:
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 5')
    gs = []
    index_layer = 0
    for i in range(0, num_layers):
        phis = thetas[i * n * 5:(i + 1) * n * 5]
        # Sub-Layer 1
        qc_copy = qtm.nqubit.create_rx_nqubit(qc.copy(), phis[:n])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(qc, phis[:n])
        index_layer += 1
        # Sub-Layer 2
        qc_copy = qtm.nqubit.create_wy(qc.copy(), phis[n:n * 2])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_wy(qc, phis[n:n * 2])
        index_layer += 1
        # Sub-Layer 3
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 2:n * 3])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 2:n * 3])
        index_layer += 1
        # Sub-Layer 4
        qc_copy = qtm.nqubit.create_wy(qc.copy(), phis[n * 3:n * 4])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_wy(qc, phis[n * 3:n * 4])
        index_layer += 1
        # Sub-Layer 5
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 4:n * 5])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 4:n * 5])
        index_layer += 1
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_alternative_layered_state(qc: qiskit.QuantumCircuit,
                                        thetas,
                                        num_layers: int = 1):
    """Create binho ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * (n * 5 - 4):
        raise Exception(
            'Number of parameters must be equal n_layers * (n*5 - 4)')
    gs = []
    index_layer = 0
    for i in range(0, num_layers):
        # Sub-Layer 1
        phis = thetas[i * (n * 5 - 4):(i + 1) * (n * 5 - 4)]
        qc_copy = qtm.nqubit.create_ry_nqubit(qc.copy(), phis[:n])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_ry_nqubit(qc, phis[:n])
        index_layer += 1
        # Sub-Layer 2
        qc = qtm.nqubit.create_swap_nqubit(qc)
        qc_copy = qtm.nqubit.create_ry_nqubit(qc.copy(), phis[n:n * 2 - 1])
        observers = (create_observers(qc_copy, k=n - 1))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_ry_nqubit(qc, phis[n:n * 2 - 1])
        index_layer += 1
        # Sub-Layer 3
        qc = qtm.nqubit.create_swap_nqubit(qc, shift=1)
        qc_copy = qtm.nqubit.create_ry_nqubit(qc.copy(),
                                              phis[n * 2 - 1:n * 3 - 2],
                                              shift=1)
        observers = (create_observers(qc_copy, k=n - 1))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_ry_nqubit(qc,
                                         phis[n * 2 - 1:n * 3 - 2],
                                         shift=1)
        index_layer += 1
        # Sub-Layer 4
        qc = qtm.nqubit.create_swap_nqubit(qc)
        qc_copy = qtm.nqubit.create_ry_nqubit(qc.copy(),
                                              phis[n * 3 - 2:n * 4 - 3])
        observers = (create_observers(qc_copy, k=n - 1))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_ry_nqubit(qc, phis[n * 3 - 2:n * 4 - 3])
        index_layer += 1
        # Sub-Layer 5
        qc = qtm.nqubit.create_swap_nqubit(qc, shift=1)
        qc_copy = qtm.nqubit.create_ry_nqubit(qc.copy(),
                                              phis[n * 4 - 3:n * 5 - 4],
                                              shift=1)
        observers = (create_observers(qc_copy, k=n - 1))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_ry_nqubit(qc,
                                         phis[n * 4 - 3:n * 5 - 4],
                                         shift=1)
        index_layer += 1

    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_Wchain_state(qc: qiskit.QuantumCircuit,
                           thetas,
                           num_layers: int = 1):
    """Create W chain ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    if len(thetas) != num_layers * (n * 4):
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 4')

    gs = []
    for i in range(0, num_layers):
        # Sub-Layer 1
        phis = thetas[i * (n * 4):(i + 1) * (n * 4)]
        qc_copy = qtm.nqubit.create_Wchain(qc.copy(), phis[:n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_Wchain(qc, phis[:n])
        # Sub-Layer 2
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n:n * 2])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n:n * 2])
        # Sub-Layer 3
        qc_copy = qtm.nqubit.create_rx_nqubit(qc.copy(), phis[n * 2:n * 3])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(qc, phis[n * 2:n * 3])
        # Sub-Layer 4
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 3:n * 4])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 3:n * 4])
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_Wchain2_state(qc: qiskit.QuantumCircuit,
                            thetas,
                            num_layers: int = 1):
    """Create W chain ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    if len(thetas) != num_layers * (n * 4):
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 4')

    gs = []
    for i in range(0, num_layers):

        phis = thetas[i * (n * 4):(i + 1) * (n * 4)]
        # Sub-Layer 1
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[:n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[:n])
        # Sub-Layer 2
        qc_copy = qtm.nqubit.create_Wchain(qc.copy(), phis[n:n * 2])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_Wchain(qc, phis[n:n * 2])
        # Sub-Layer 3
        qc_copy = qtm.nqubit.create_rx_nqubit(qc.copy(), phis[n * 2:n * 3])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(qc, phis[n * 2:n * 3])
        # Sub-Layer 4
        qc_copy = qtm.nqubit.create_rz_nqubit(qc.copy(), phis[n * 3:n * 4])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n * 3:n * 4])
    G = gs[0]

    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_Walternating_state(qc: qiskit.QuantumCircuit,
                                 thetas,
                                 num_layers: int = 1):
    """Create W alternating ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    n_param = 0
    gs = []

    for i in range(0, num_layers):
        n_alternating = qtm.nqubit.calculate_n_walternating(i, n)
        phis = thetas[n_param:n_param + n_alternating + 3 * n]
        n_param += n_alternating + 3 * n
        # Sub-layer 1
        qc_copy = qtm.nqubit.create_Walternating(qc.copy(),
                                                 phis[:n_alternating], i + 1)
        observers = create_observers(qc_copy, k=n_alternating)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_Walternating(qc.copy(), phis[:n_alternating],
                                            i + 1)
        # print(qc.draw())
        # Sub-layer 2
        qc_copy = qtm.nqubit.create_rz_nqubit(
            qc.copy(), phis[n_alternating:n_alternating + n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc,
                                         phis[n_alternating:n_alternating + n])
        # Sub-layer 3
        qc_copy = qtm.nqubit.create_rx_nqubit(
            qc.copy(), phis[n_alternating + n:n_alternating + n * 2])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(
            qc, phis[n_alternating + n:n_alternating + n * 2])
        # Sub-layer 4
        qc_copy = qtm.nqubit.create_rz_nqubit(
            qc.copy(), phis[n_alternating + n * 2:n_alternating + n * 3])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(
            qc, phis[n_alternating + n * 2:n_alternating + n * 3])
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_Walltoall_state(qc: qiskit.QuantumCircuit,
                              thetas,
                              num_layers: int = 1):
    """Create W alltoall ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    n_walltoall = qtm.nqubit.calculate_n_walltoall(n)

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * 3 * n + num_layers * n_walltoall:
        raise Exception(
            'The number of parameter must be equal num_layers* 3 * n + num_layers*n_walltoall'
        )
    n_param = 0
    gs = []
    index_layer = 0
    for i in range(0, num_layers):

        phis = thetas[i * (3 * n) + i * n_walltoall:(i + 1) * (3 * n) +
                      (i + 1) * n_walltoall]
        # Sub-layer 1 -> n - 1
        num_observers = list(range(1, n))
        num_observers.reverse()
        limit = 0
        for num_observer in num_observers:
            limit += num_observer
            qc_copy = qtm.nqubit.create_Walltoall(qc.copy(),
                                                  phis[:n_walltoall],
                                                  limit=limit)
            observers = create_observers(qc_copy, k=num_observer)
            gs.append(calculate_g(qc_copy, observers))

        qc = qtm.nqubit.create_Walltoall(qc, phis[:n_walltoall])
        index_layer += 1

        # Sub-layer 2
        qc_copy = qtm.nqubit.create_rz_nqubit(
            qc.copy(), phis[n_walltoall:n_walltoall + n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(qc, phis[n_walltoall:n_walltoall + n])

        index_layer += 1
        # Sub-layer 3
        qc_copy = qtm.nqubit.create_rx_nqubit(
            qc.copy(), phis[n_walltoall + n:n_walltoall + 2 * n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(
            qc, phis[n_walltoall + n:n_walltoall + 2 * n])
        index_layer += 1
        # Sub-layer 4
        qc_copy = qtm.nqubit.create_rz_nqubit(
            qc.copy(), phis[n_walltoall + 2 * n:n_walltoall + 3 * n])
        observers = create_observers(qc_copy)
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rz_nqubit(
            qc, phis[n_walltoall + 2 * n:n_walltoall + 3 * n])
        index_layer += 1

    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G


def calculate_star2graph_state(qc: qiskit.QuantumCircuit,
                               thetas,
                               num_layers: int = 1):
    """Create W alltoall ansatz and compuate g each sub-layer

    Args:
        qc (qiskit.QuantumCircuit): Init circuit (blank)
        thetas (Numpy array): Parameters
        n_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers*(2*n - 2):
        raise Exception(
            'The number of parameter must be equal num_layers * (2 * num_qubits - 2)'
        )
    gs = []
    index_layer = 0
    j = 0
    for i in range(0, num_layers):
        phis = thetas[i * n * 5:(i + 1) * n * 5]
        # Sub-Layer 1
        qc_copy = qtm.nqubit.create_rx_nqubit(qc.copy(), phis[:n])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_rx_nqubit(qc, phis[:n])
        index_layer += 1
        # Sub-Layer 2
        qc_copy = qtm.nqubit.create_cry_nqubit_inverse(qc.copy(),
                                                       phis[n:n * 2])
        observers = (create_observers(qc_copy))
        gs.append(calculate_g(qc, observers))
        qc = qtm.nqubit.create_cry_nqubit_inverse(qc, phis[n:n * 2])
        index_layer += 1
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G

######################################
## General quantum natural gradient ##
######################################


def get_wires_of_gate(gate: Tuple):
    """Get index bit that gate act on

    Args:
        gate (qiskit.QuantumGate): Quantum gate

    Returns:
        numpy arrray: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register.index)
    return list_wire


def is_gate_in_list_wires(gate: Tuple, wires: List):
    """Check if a gate lies on the next layer or not

    Args:
        gate (qiskit.QuantumGate): Quantum gate
        wires (numpy arrray): list of index bits

    Returns:
        Bool
    """
    list_wire = get_wires_of_gate(gate)
    for wire in list_wire:
        if wire in wires:
            return True
    return False


def split_into_layers(qc: qiskit.QuantumCircuit):
    """Split a quantum circuit into layers

    Args:
        qc (qiskit.QuantumCircuit): origin circuit

    Returns:
        list: list of list of quantum gates
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
        if len(param) == 0:
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


def add_layer_into_circuit(qc: qiskit.QuantumCircuit, layer: List):
    """Based on information in layer, adding new gates into current circuit

    Args:
        qc (qiskit.QuantumCircuit): calculating circuit
        layer (list): list of gate's informations

    Returns:
        qiskit.QuantumCircuit: added circuit
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


def qng(qc: qiskit.QuantumCircuit, thetas, create_circuit_func: FunctionType, num_layers: int):
    n = qc.num_qubits
    # List of g matrices
    gs = []
    # Temporary circuit
    qc_new = qiskit.QuantumCircuit(n, n)
    qc_new = create_circuit_func(qc_new, thetas, num_layers)
    # Splitting circuit into list of V and W sub-layer (non-parameter and parameter)
    layers = split_into_layers(qc_new)
    if num_layers == 1:
        for is_param_layer, layer in layers:
            if is_param_layer:
                observers = qtm.fubini_study.create_observers(
                    add_layer_into_circuit(qc.copy(), layer), len(layer))
                gs.append(qtm.fubini_study.calculate_g(qc, observers))
            # Add next sub-layer into the current circuit
            qc = add_layer_into_circuit(qc, layer)
    else:
        for i in range(0, num_layers):
            for is_param_layer, layer in layers[i * int(len(layers) / 2): (i + 1) * int(len(layers) / 2)]:
                if is_param_layer:
                    observers = qtm.fubini_study.create_observers(
                        add_layer_into_circuit(qc.copy(), layer), len(layer))
                    gs.append(qtm.fubini_study.calculate_g(qc, observers))
                # Add next sub-layer into the current circuit
                qc = add_layer_into_circuit(qc, layer)
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G

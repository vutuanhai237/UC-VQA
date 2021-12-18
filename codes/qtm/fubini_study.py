import qiskit
import numpy as np
import qtm.constant
from typing import Dict
from scipy.linalg import block_diag


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
            wire = qc.num_qubits - 1 - gate[1][1].index
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
            # # Ignore noise
            # if g[i, j] < 10**(-15):
            #     g[i, j] = 0
    return g


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
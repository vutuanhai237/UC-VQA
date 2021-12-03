import qiskit
import numpy as np
import qtm.constant
from typing import Dict
from scipy.linalg import block_diag

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
    psi = np.expand_dims(psi , 1)
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
            g[i, j] = psi_hat @ (Ks[i] @ Ks[j]) @ psi - (psi_hat @ Ks[i] @ psi)*(psi_hat @ Ks[j] @ psi)
            # Ignore noise
            if g[i, j] < 10**(-15):
                g[i, j] = 0
    return g

def calculate_koczor_state(qc: qiskit.QuantumCircuit, thetas, num_layers: int = 1):
    """Create koczor anzsats and compuate g each sub-layer

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
        raise Exception('Number of parameters must be equal n_layers * num_qubits * 5')
    gs = []
    index_layer = 0
    for i in range(0, num_layers):
        phis = thetas[i:(i + 1)*n*5]
        qc_copy = qtm.qtm_nqubit.create_rx_nqubit(qc.copy(), phis[:n])
        observers = (qtm.base_qtm.create_observers(qc_copy))[index_layer]
        gs.append(calculate_g(qc, observers))
        qc = qtm.qtm_nqubit.create_rx_nqubit(qc, phis[:n])
        index_layer += 1

        qc_copy = qtm.qtm_nqubit.create_cry_nqubit_inverse(qc.copy(), phis[n:n*2])
        observers = (qtm.base_qtm.create_observers(qc_copy))[index_layer]
        gs.append(calculate_g(qc, observers))
        qc = qtm.qtm_nqubit.create_cry_nqubit_inverse(qc, phis[n:n*2])
        index_layer += 1


        qc_copy = qtm.qtm_nqubit.create_rz_nqubit(qc.copy(), phis[n*2:n*3])
        observers = (qtm.base_qtm.create_observers(qc_copy))[index_layer]
        gs.append(calculate_g(qc, observers))
        qc = qtm.qtm_nqubit.create_rz_nqubit(qc, phis[n*2:n*3])
        index_layer += 1


        qc_copy = qtm.qtm_nqubit.create_cry_nqubit(qc.copy(), phis[n*3:n*4])
        observers = (qtm.base_qtm.create_observers(qc_copy))[index_layer]
        gs.append(calculate_g(qc, observers))
        qc = qtm.qtm_nqubit.create_cry_nqubit(qc, phis[n*3:n*4])
        index_layer += 1


        qc_copy = qtm.qtm_nqubit.create_rz_nqubit(qc.copy(), phis[n*4:n*5])
        observers = (qtm.base_qtm.create_observers(qc_copy))[index_layer]
        gs.append(calculate_g(qc, observers))
        qc = qtm.qtm_nqubit.create_rz_nqubit(qc, phis[n*4:n*5])
        index_layer += 1
    G = gs[0]
    for i in range(1, len(gs)):
        G = block_diag(G, gs[i])
    return G
import qiskit
import numpy as np
import qtm.constant
from typing import Dict

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
            K = qtm.constant.generator['I']
        for i in range(1, num_qubits):
            if i == observer_wire:
                K = np.kron(K, observer)
            else:
                K = np.kron(K, qtm.constant.generator['I']) 
        Ks.append(K)

    for i in range(0, num_observers):
        for j in range(0, num_observers):
            g[i, j] = psi_hat @ (Ks[i] @ Ks[j]) @ psi - (psi_hat @ Ks[i] @ psi)*(psi_hat @ Ks[j] @ psi)
            # Ignore noise
            if g[i, j] < 10**(-15):
                g[i, j] = 0
    return g
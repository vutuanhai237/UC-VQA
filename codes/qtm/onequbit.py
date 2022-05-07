import qiskit
import numpy as np

def u_onequbit(qc: qiskit.QuantumCircuit, thetas: np.ndarray, wire: int = 0):
    """Return a simple series of 1 qubit gate

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - wire (int): position that the gate carries on

    Returns:
        - QuantumCircuit: The circuit which have added gates
    """
    if isinstance(wire, int) != True:
        wire = (wire['wire'])
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    return qc


def u_onequbit_h(qc: qiskit.QuantumCircuit, thetas: np.ndarray, wire: int):
    """Return a simple series of 1 qubit - gate which is measured in X-basis

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - wire (int): position that the gate carries on   

    Returns:
        - QuantumCircuit: The circuit which have added gates
    """
    if isinstance(wire, int) != True:
        wire = (wire['wire'])
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    qc.h(wire)
    return qc

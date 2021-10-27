from types import FunctionType
import qiskit
import qtm.base_qtm
import numpy as np

def u_1q(qc: qiskit.QuantumCircuit, thetas, wire: int = 0):
    """Return a simple series of 1 qubit gate

    Args:
        qc (QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
        wire (Int): position that the gate carries on

    Returns:
        QuantumCircuit: The circuit which have added gates
    """
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    return qc

def u_1q_h(qc: qiskit.QuantumCircuit, thetas, wire: int = 0):
    """Return a simple series of 1qubit - gate which is measured in X-basis

    Args:
        qc (QuantumCircuit): Init circuit
        thetas (Numpy array): Parameters
        wire (Int): position that the gate carries on   

    Returns:
        QuantumCircuit: The circuit which have added gates
    """
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    qc.h(wire)
    return qc

def get_u_1q_hat(thetas, num_qubits: int = 1):
    """Return inverse of u_1q gate

    Args:
        thetas (Numpy array): Parameters
        num_qubits (Int): number of qubit
    Returns:
        Statevector: The state vector of when applying u_1q gate
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = u_1q(qc, thetas).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)

def get_u_1q_h_hat(thetas, num_qubits: int = 1):
    """Return inverse of u_1q_h gate

    Args:
        thetas (Numpy array): Parameters
        num_qubits (Int): number of qubit
    Returns:
        Statevector: The state vector of when applying u_1q_h gate
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = u_1q_h(qc, thetas).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)


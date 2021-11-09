import qiskit
import numpy as np
def cf(qc: qiskit.QuantumCircuit, theta: float, qubit1: int, qubit2: int):
    """Add Controlled-F gate to quantum circuit

    Args:
        qc (qiskit.QuantumCircuit): Added circuit
        theta (float): = arccos(1/sqrt(num_qubits), base on number of qubit
        qubit1 (int): control qubit
        qubit2 (int): target qubit

    Returns:
        qiskit.QuantumCircuit: Added circuit
    """
    cf = qiskit.QuantumCircuit(2)
    u = np.array([
        [1, 0, 0, 0], 
        [0, np.cos(theta), 0, np.sin(theta)],
        [0, 0, 1, 0],
        [0, np.sin(theta), 0, -np.cos(theta)]
    ])
    cf.unitary(u, [0, 1])
    cf_gate = cf.to_gate(label = 'CF')  
    qc.append(cf_gate, [qubit1, qubit2])
    return qc

def w3(circuit: qiskit.QuantumCircuit, qubit: int):
    """Create W state for 3 qubits

    Args:
        circuit (qiskit.QuantumCircuit): Added circuit
        qubit (int): the index that w3 circuit acts on

    Returns:
        qiskit.QuantumCircuit: Added circuit
    """
    qc = qiskit.QuantumCircuit(3)
    theta = np.arccos(1/np.sqrt(3))
    qc.cf(theta, 0, 1)
    qc.cx(1, 0)
    qc.ch(1, 2)
    qc.cx(2, 1)
    w3 = qc.to_gate(label = 'w3')
    # Add the gate to your circuit which is passed as the first argument to cf function:
    circuit.append(w3, [qubit, qubit + 1, qubit + 2])
    return circuit

qiskit.QuantumCircuit.w3 = w3
qiskit.QuantumCircuit.cf = cf
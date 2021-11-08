import qiskit
import numpy as np
def cf(qc, qubit1, qubit2):
    cf = qiskit.QuantumCircuit(2)
    theta = np.arccos(1/np.sqrt(qc.num_qubits))
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
qiskit.QuantumCircuit.cf = cf
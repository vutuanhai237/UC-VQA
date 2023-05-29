import qiskit
import numpy as np

# Training hyperparameter
num_shots = 10000
learning_rate = 0.1
noise_prob = 0.04 # [0, 1]
gamma = 0.7 # learning rate decay rate
delta = 0.01 # minimum change value of loss value
discounting_factor = 0.3 # [0, 1]
backend = qiskit.Aer.get_backend('qasm_simulator')

# For parameter-shift rule
two_term_psr = {
    'r': 1/2,
    's': np.pi / 2
}

four_term_psr = {
    'alpha': np.pi / 2,
    'beta' : 3 * np.pi / 2,
    'd_plus' : (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}

one_qubit_gates = ["Hadamard", 'RX', 'RY', 'RZ']
two_qubits_gates = ['CNOT', 'CY', 'CZ', 'CRX', 'CRY', 'CRZ']

def create_gate_pool(num_qubits, one_qubit_gates = one_qubit_gates, two_qubits_gates = two_qubits_gates):
    gate_pool = []

    # Single-qubit gates
    single_qubit_gates = one_qubit_gates
    for qubit in range(num_qubits):
        for gate in single_qubit_gates:
            gate_pool.append((gate, qubit))

    # Two-qubit gates
    two_qubit_gates = two_qubits_gates
    for qubit1 in range(num_qubits):
        for qubit2 in range(num_qubits):
            if qubit1 != qubit2:
                for gate in two_qubit_gates:
                    gate_pool.append((gate, qubit1, qubit2))

    return gate_pool

# For QNG
generator = {
    'cu': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'crx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'ry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'cry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'crz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'cz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'i': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'id': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    '11': np.array([[0, 0], [0, 1]], dtype=np.complex128),
}

ignore_generator = [
    'barrier'
]
parameterized_generator = [
    'rx', 'ry', 'rz', 'crx', 'cry', 'crz'
]

# This information is extracted from http://dx.doi.org/10.1103/PhysRevA.83.042314
edges_graph_state = {
    2: ["0-1"],
    3: ["0-1", "0-2"],
    4: ["0-2", "1-3", "2-3"],
    5: ["0-3", "2-4", "1-4", "2-3"],
    6: ["0-4", "1-5", "2-3", "2-4", "3-5"],
    7: ["0-5", "2-3", "4-6", "1-6", "2-4", "3-5"],
    8: ["0-7", "1-6", "2-4", "3-5", "2-3", "4-6", "5-7"],
    9: ["0-8", "2-3", "4-6", "5-7", "1-7", "2-4", "3-5", "6-8"],
    10: ["0-9", "1-8", "2-3", "4-6", "5-7", "2-4", "3-5", "6-9", "7-8"]
}

look_up_operator = {
    "Identity": 'I',
    "Hadamard": 'H',
    "PauliX": 'X',
    'PauliY': 'Y',
    'PauliZ': 'Z',
    'S': 'S',
    'T': 'T',
    'SX': 'SX',
    'CNOT': 'CX',
    'CZ': 'CZ',
    'CY': 'CY',
    'SWAP': 'SWAP',
    'ISWAP': 'ISWAP',
    'CSWAP': 'CSWAP',
    'Toffoli': 'CCX',
    'RX': 'RX',
    'RY': 'RY',
    'RZ': 'RZ',
    'CRX': 'CRX',
    'CRY': 'CRY',
    'CRZ': 'CRZ',
    'U1': 'U1',
    'U2': 'U2',
    'U3': 'U3',
    'IsingXX': 'RXX',
    'IsingYY': 'RYY',
    'IsingZZ': 'RZZ',
}

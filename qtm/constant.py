import qiskit
import numpy as np
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate,
                                                   YGate, ZGate, HGate, SGate, SdgGate, TGate,
                                                   TdgGate, RXGate, RYGate, RZGate, CXGate,
                                                   CYGate, CZGate, CHGate, CRXGate, CRYGate, CRZGate, CU1Gate,
                                                   CU3Gate, SwapGate, RZZGate,
                                                   CCXGate, CSwapGate)

# Training hyperparameter
num_shots = 10000
learning_rate = 0.1
noise_prob = 0.0 # [0, 1]
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



H_gate = {'name': 'h', 'operation': HGate, 'num_op': 1, 'num_params': 0}
S_gate = {'name': 's', 'operation': SGate, 'num_op': 1, 'num_params': 0}
X_gate = {'name': 'x', 'operation': XGate, 'num_op': 1, 'num_params': 0}
Y_gate = {'name': 'y', 'operation': YGate, 'num_op': 1, 'num_params': 0}
Z_gate = {'name': 'z', 'operation': ZGate, 'num_op': 1, 'num_params': 0}
CX_gate = {'name': 'cx', 'operation': CXGate, 'num_op': 2, 'num_params': 0}
CRX_gate = {'name': 'crx', 'operation': CRXGate, 'num_op': 2, 'num_params': 1}
CRY_gate = {'name': 'cry', 'operation': CRYGate, 'num_op': 2, 'num_params': 1}
CRZ_gate = {'name': 'crz', 'operation': CRZGate, 'num_op': 2, 'num_params': 1}
RX_gate = {'name': 'rx', 'operation': RXGate, 'num_op': 1, 'num_params': 1}
RY_gate = {'name': 'ry', 'operation': RYGate, 'num_op': 1, 'num_params': 1}
RZ_gate = {'name': 'rz', 'operation': RZGate, 'num_op': 1, 'num_params': 1}
U2_gate = {'name': 'u2', 'operation': U2Gate, 'num_op': 1, 'num_params': 2}
U3_gate = {'name': 'u3', 'operation': U3Gate, 'num_op': 1, 'num_params': 3}

clifford_set = [
    H_gate,
    CX_gate,
    S_gate
    
]


operations = [
    H_gate,
    S_gate,
    X_gate,
    Y_gate,
    Z_gate,
    CX_gate,
    RX_gate,
    RY_gate,
    RZ_gate,
    CRX_gate,
    CRY_gate,
    CRZ_gate
]



one_q_ops_name = ['h','rx','ry','rz', 'cx']
one_q_ops = [HGate, RXGate, RYGate, RZGate]
one_param_name = ['rx','ry','rz']
one_param = [RXGate, RYGate, RZGate]
two_param_name = ['u2']
two_param = [U2Gate]
two_q_ops = [CXGate]
three_param_name = ['u3']
three_param = [U3Gate]
three_q_ops_name = ['ccx']
three_q_ops = [CCXGate]

# one_q_ops = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
# two_q_ops = [CXGate, CYGate, CZGate, CHGate, CRZGate,
#              CU1Gate, CU3Gate, SwapGate, RZZGate]
# one_q_ops = [RXGate, RYGate, RZGate]


# one_q_ops = [XGate, YGate, ZGate, HGate, RXGate, RYGate, RZGate]

# one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
# two_param = [U2Gate]
# three_param = [U3Gate, CU3Gate]

# two_q_ops = [CXGate, CRZGate]
# three_q_ops = [CCXGate, CSwapGate]
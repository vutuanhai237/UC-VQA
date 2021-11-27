import qiskit
import numpy as np
num_shots = 10000
learning_rate = 0.2
backend = qiskit.Aer.get_backend('qasm_simulator')

generator = {
    'RX': -1/2*np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.complex128),
    'RY': -1/2*np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=np.complex128),
    'RZ': -1/2*np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.complex128),
    'I': np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.complex128)
}
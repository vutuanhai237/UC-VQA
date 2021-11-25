import qiskit
import numpy as np
num_shots = 8192
learning_rate = 0.2
backend = qiskit.Aer.get_backend('qasm_simulator')

generator = {
    'rx': 1/2*np.array([
        [0, 1],
        [1, 0]
    ]),
    'ry': 1/2*np.array([
        [0, 1],
        [1, 0]
    ]),
    'rx': 1/2*np.array([
        [0, 1],
        [1, 0]
    ]),
}
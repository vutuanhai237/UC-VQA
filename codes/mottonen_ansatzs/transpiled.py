import pennylane as qml
import numpy as np
from pennylane_qiskit import AerDevice
dev = AerDevice(wires=4)
dev = qml.device('default.qubit', wires=3)
@qml.qnode(dev)
def circuit(state):
    qml.MottonenStatePreparation(state_vector=state, wires=range(3))
    return qml.state()

state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
state = state / np.linalg.norm(state)
circuit(state)

dev._circuit.qasm(formatted=True)
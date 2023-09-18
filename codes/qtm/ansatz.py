import qiskit
import math
from qtm.utilities import compose_circuit
import qtm.base
import qtm.state
import numpy as np
import random


def graph_ansatz(num_qubits) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    edges = qtm.constant.edges_graph_state[num_qubits]
    thetas = qiskit.circuit.ParameterVector('theta', len(edges))
    i = 0
    for edge in edges:
        control_bit = int(edge.split('-')[0])
        controlled_bit = int(edge.split('-')[1])
        qc.crz(thetas[i], control_bit, controlled_bit)
        i += 1
    return qc


def stargraph_ansatz(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:

    thetas = qiskit.circuit.ParameterVector(
        'theta', num_layers * (2 * num_qubits - 2))
    qc = qiskit.QuantumCircuit(num_qubits)
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(2, num_qubits):
            qc.ry(thetas[j], 0)
            j += 1
            qc.cz(0, i)
    return qc


def polygongraph_ansatz(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', 2*num_qubits*num_layers)

    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            for i in range(0, num_qubits - 1):
                qc.ry(thetas[j], i)
                j += 1
        else:
            for i in range(0, num_qubits):
                qc.ry(thetas[j], i)
                j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            qc.ry(thetas[j], num_qubits - 1)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc


def hadamard_hypergraph_ansatz(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', 3*num_qubits*num_layers)
    j = 0
    for i in range(0, num_qubits):
        qc.h(i)
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, num_qubits - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], num_qubits - 1)
        j += 1
        qc.cz(num_qubits - 2, num_qubits - 1)
        for i in range(1, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0, 1, 2)
    return qc


def hypergraph_ansatz(num_qubits: int, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', 3*num_qubits*num_layers)
    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, num_qubits - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], num_qubits - 1)
        j += 1
        qc.cz(num_qubits - 2, num_qubits - 1)
        for i in range(1, num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0, 1, 2)
    return qc


def hypergraph_w_ansatz(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit(
            [qc, hypergraph_ansatz(num_qubits), zxz_layer(num_qubits)])
    return qc


def entangled_layer(qc: qiskit.QuantumCircuit):
    """Create entanglement state

    Args:
        - qc (QuantumCircuit): init circuit

    Returns:
        - QuantumCircuit: the added circuit
    """
    for i in range(0, qc.num_qubits):
        if i == qc.num_qubits - 1:
            qc.cnot(qc.num_qubits - 1, 0)
        else:
            qc.cnot(i, i + 1)
    return qc


def cry_layer(num_qubits) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits)
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    for i in range(1, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[qc.num_qubits - 1], qc.num_qubits - 1, 0)
    return qc


def wy_layer(num_qubits: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', (num_qubits + 1))
    for i in range(0, num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    for i in range(1, num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    qc.cry(thetas[num_qubits - 1], 0, num_qubits - 1)
    return qc


def binho_ansatz(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, rx_layer(num_qubits),
                              wy_layer(num_qubits),
                              rz_layer(num_qubits),
                              wy_layer(num_qubits),
                              rz_layer(num_qubits)])
    return qc


def ry_layer(num_qubits: int = 3, shift=0) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits - shift)
    for i in range(0, num_qubits):
        qc.ry(thetas[i], i + shift)
    return qc


def swap_layer(num_qubits: int = 3, shift=0) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for i in range(0 + shift, qc.num_qubits - 1, 2):
        qc.swap(i, i + 1)
    return qc


def alternating_ZXZlayer_ansatz(num_qubits: int = 3,
                                num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, ry_layer(num_qubits),
                              swap_layer(num_qubits),
                              ry_layer(num_qubits),
                              swap_layer(num_qubits, shift=1),
                              ry_layer(num_qubits, shift=1),
                              swap_layer(num_qubits),
                              ry_layer(num_qubits),
                              swap_layer(num_qubits, shift=1),
                              ry_layer(num_qubits, shift=1)])
    return qc


###########################
#### Tomography circuit ###
###########################


def Wchain(num_qubits: int) -> qiskit.QuantumCircuit:
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits)
    for i in range(0, num_qubits - 1):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[-1], num_qubits - 1, 0)
    return qc


def Walternating(num_qubits: int, thetas: np.ndarray, index_layer) -> qiskit.QuantumCircuit:

    def calculate_n_walternating(num_qubits: int, index_layers: int):
        if index_layers % 2 == 0:
            n_walternating = int(num_qubits / 2)
        else:
            n_walternating = math.ceil(num_qubits / 2)

        return n_walternating

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector(
        'theta', calculate_n_walternating(num_qubits, index_layer))
    if index_layer % 2 == 0:
        # Even
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
        qc.cry(thetas[-1], 0, qc.num_qubits - 1)
    else:
        # Odd
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
    return qc


def WalternatingCNOT(qc: qiskit.QuantumCircuit, index_layer) -> qiskit.QuantumCircuit:
    t = 0
    if index_layer % 2 == 0:
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cnot(i, i + 1)
            t += 1
        qc.cnot(0, qc.num_qubits - 1)
    else:
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cnot(i, i + 1)
            t += 1
    return qc


def Walltoall(num_qubits: int, limit=0) -> qiskit.QuantumCircuit:

    def calculate_n_walltoall(num_qubits):
        n_walltoall = 0
        for i in range(1, num_qubits):
            n_walltoall += i
        return n_walltoall

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector(
        'theta', calculate_n_walltoall(num_qubits))
    if limit == 0:
        limit = len(thetas)
    t = 0
    for i in range(0, num_qubits):
        for j in range(i + 1, num_qubits):
            qc.cry(thetas[t], i, j)
            t += 1
            if t == limit:
                return qc
    return qc


def WalltoallCNOT(qc: qiskit.QuantumCircuit, limit=0) -> qiskit.QuantumCircuit:
    """Create Walltoall CNOT

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - limit (int): limit layer

    Returns:
        - qiskit.QuantumCircuit
    """
    t = 0
    for i in range(0, qc.num_qubits):
        for j in range(i + 1, qc.num_qubits):
            qc.cnot(i, j)
            t += 1
            if t == limit:
                return qc
    return qc


def Wchain_ZXZlayer_ansatz(num_qubits, num_layers: int = 1) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        num_qubits (_type_): _description_
        num_layers (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, Wchain(num_qubits),
                              zxz_layer(num_qubits)])
    return qc


def Walternating_ZXZlayer_ansatz(num_qubits, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, Wchain(num_qubits),
                              zxz_layer(num_qubits)])
    return qc


def WalternatingCNOT_ZXZlayer_ansatz(num_qubits, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, WalternatingCNOT(num_qubits),
                              zxz_layer(num_qubits)])
    return qc


def Walltoall_ZXZlayer_ansatz(num_qubits, num_layers: int = 1, limit=0) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit(
            [qc, Walltoall(qc, limit=limit), zxz_layer(num_qubits)])
    return qc


def WalltoallCNOT_ZXZlayer_ansatz(num_qubits: int = 3,
                                  num_layers: int = 1,
                                  limit=0) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit(
            [qc, WalltoallCNOT(qc, limit=limit), zxz_layer(num_qubits)])
    return qc


def zxz_layer(num_qubits: int = 3, num_layers: int = 1) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit(
            [qc, rz_layer(num_qubits), rx_layer(num_qubits), rz_layer(num_qubits)])
    return qc


def random_ccz_circuit(num_qubits, num_gates) -> qiskit.QuantumCircuit:
    """Adds a random number of CZ or CCZ gates (up to `max_gates`) to the given circuit."""
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    sure = True if num_gates < 3 else False
    for _ in range(num_gates):
        if np.random.randint(2, size=1) == 0 or sure:
            wires = random.sample(range(0, num_qubits), 2)
            qc.cz(wires[0], wires[1])
        else:
            wires = random.sample(range(0, num_qubits), 3)
            wires.sort()
            qc.ccz(wires[0], wires[1], wires[2])
    return qc


def rz_layer(num_qubits) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.rz(thetas[i], i)
    return qc


def rx_layer(num_qubits) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.rx(thetas[i], i)
    return qc


def ry_layer(num_qubits) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector('theta', num_qubits)
    for i in range(num_qubits):
        qc.ry(thetas[i], i)
    return qc


def cz_layer(num_qubits) -> qiskit.QuantumCircuit:
    return qiskit.circuit.library.MCMT('z', num_qubits - 1, 1)


def g2(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    thetas = qiskit.circuit.ParameterVector(
        'theta', 2 * num_qubits * num_layers)
    j = 0
    for i in range(num_qubits):
        qc.ry(thetas[j], i)
        j += 1
    for i in range(0, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    for i in range(num_qubits):
        qc.ry(thetas[j], i)
        j += 1
    for i in range(1, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    qc.cz(0, num_qubits - 1)
    return qc


def gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, ry_layer(num_qubits), cz_layer(num_qubits)])
    return qc


def g2gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, g2(num_qubits, 1), gn(num_qubits, 1)])
    return qc


def g2gnw(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 + gn + w ansatz

    Args:
        num_qubits (int): _description_
        num_layers (int): _description_

    Returns:
        qiskit.QuantumCircuit: _description_
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, g2(num_qubits, 1), gn(
            num_qubits, 1), zxz_layer(num_qubits, 1)])
    return qc

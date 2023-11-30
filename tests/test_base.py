
import qiskit
import numpy as np
import sys
import os
sys.path.insert(0, '../')
from qsee.core import state, ansatz
from qsee.compilation.qcompilation import QuantumCompilation
from qsee.compilation.qsp import QuantumStatePreparation
def test_onequbit_qst():
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    lambdaz = 0
    qcu3 = qiskit.QuantumCircuit(1, 1)
    qcu3.u(theta, phi, lambdaz, 0)
    compiler = QuantumCompilation(
        u=qcu3,
        vdagger=ansatz.zxz_layer(1).inverse(),
    ).fit()
    assert (np.min(compiler.metrics['loss_fubini_study']) < 0.0001)

def test_nqubit_qsp():
    num_qubits = 3
    num_layers = 2
    compiler = QuantumCompilation(
        u = ansatz.g2gn(num_qubits, num_layers),
        vdagger = state.ghz(num_qubits).inverse(),
    ).fit()
    assert (np.min(compiler.metrics['loss_fubini_study']) < 0.0001)
def test_nqubit_qst():
    num_qubits = 3
    num_layers = 1
    compiler = QuantumCompilation(
        u=state.ghz(num_qubits),
        vdagger=ansatz.Wchain_zxz(num_qubits, num_layers),
    ).fit()
    assert (np.min(compiler.metrics['loss_fubini_study']) < 0.0001)

def test_save_load_qsp():
    num_qubits = 3
    num_layers = 1
    qspobj = QuantumStatePreparation(
        u = ansatz.g2gn(num_qubits, num_layers),
        target_state = state.ghz(num_qubits).inverse(),
    )
    qspobj.fit()
    qspobj.save('haar')
    qspobj2 = QuantumStatePreparation.load('./haar')
    import shutil
    shutil.rmtree('./haar')
    assert qspobj2.fidelity > 0.99
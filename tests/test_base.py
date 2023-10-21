
import qiskit
import numpy as np
import sys
import os
sys.path.insert(0, '..')
import qtm.ansatz
import qtm.qcompilation
import qtm.qsp
import qtm.state
def test_onequbit_qst():
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    lambdaz = 0

    qcu3 = qiskit.QuantumCircuit(1, 1)
    qcu3.u(theta, phi, lambdaz, 0)
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qcu3,
        vdagger=qtm.ansatz.zxz_layer(1).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=100, verbose=1)
    assert (np.min(compiler.loss_values) < 0.0001)

def test_nqubit_qsp():
    num_qubits = 3
    num_layers = 2
    optimizer = 'adam'
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.ansatz.g2gn(num_qubits, num_layers),
        vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
        optimizer = optimizer,
        loss_func = 'loss_fubini_study'
    )
    compiler.fit(num_steps = 100, verbose = 1)
    assert (np.min(compiler.loss_values) < 0.0001)
def test_nqubit_qst():
    num_qubits = 3
    num_layers = 1
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qtm.state.create_ghz_state(num_qubits),
        vdagger=qtm.ansatz.Wchain_ZXZlayer_ansatz(num_qubits, num_layers),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=100, verbose=1)
    assert (np.min(compiler.loss_values) < 0.0001)

def test_save_load_qsp():
    num_qubits = 3
    num_layers = 1
    optimizer = 'adam'
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.ansatz.g2gn(num_qubits, num_layers),
        vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
        optimizer = optimizer,
        loss_func = 'loss_fubini_study'
    )
    compiler.fit(num_steps = 100, verbose = 1)
    compiler.save(qtm.ansatz.g2gn, 'haar', './')
    qspobj2 = qtm.qsp.QuantumStatePreparation('./haar_g2gn_3_1.qspobj')
    fidelity = qspobj2.fidelity
    os.remove('haar_g2gn_3_1.qspobj')
    assert fidelity > 0.99
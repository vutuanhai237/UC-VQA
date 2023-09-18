import pytest
import person

import qiskit
import numpy as np
import codes.qtm as qtm

def onequbit_tomography():
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    lambdaz = 0

    qcu3 = qiskit.QuantumCircuit(1, 1)
    qcu3.u(theta, phi, lambdaz, 0)
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qcu3,
        vdagger = qtm.ansatz.zxz_layer(1).inverse(),
        optimizer = 'adam',
        loss_func = 'loss_fubini_study'
    )
    compiler.fit(num_steps = 100, verbose = 1)
    assert (np.min(compiler.loss_values) < 0.0001)
    
def testa():
    assert 5 == person.sum(1, 4)

def test_less():
    num = 100
    assert num < 200

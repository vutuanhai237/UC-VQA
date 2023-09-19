import qiskit     
import numpy as np
import sys
sys.path.insert(0, '..')
import qtm.qcompilation, qtm.ansatz

def test_onequbit_tomography():
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


test_onequbit_tomography()

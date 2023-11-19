import qiskit, sys
import numpy as np
sys.path.insert(1, '../../')
import qsee.compilation.qcompilation, qsee.ansatz


def run_wchain(num_layers, num_qubits):
    thetas = np.ones(num_layers*num_qubits*4)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    u = qiskit.QuantumCircuit(num_qubits, num_qubits)
    u.initialize(psi, range(0, num_qubits))

    compiler = qsee.compilation.qcompilation.QuantumCompilation(
        u = u,
        vdagger = qsee.ansatz.create_Wchain_layered_ansatz,
        optimizer = 'qng-fubini-study',
        loss_func = 'loss-fubini-study',
        thetas = thetas,
        num_layers = num_layers
    )

    compiler.fit(num_steps=400, verbose = 1)
    print('Writting ... ' + str(num_layers) + ' layers,' + str(num_qubits) +
          ' qubits')

    compiler.save(text=str(qsee.constant.noise_prob) + "mitigating", save_all = True)

run_wchain(2, 5)
print("Done!")
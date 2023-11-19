import qiskit
import numpy as np
import sys
sys.path.insert(1, '../../')
import qsee.measure, qsee.backend.constant, qsee.gradient
layers = range(6, 9)
ts = []
for num_layers in layers:
    variances = []
    grads = []
    num_qubits = 4
    thetas = np.ones(num_layers*num_qubits*4)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(0, num_qubits))
    loss_values = []
    thetass = []
    for i in range(0, 200):
        if i % 20 == 0:
            print('W_chain: (' + str(num_layers) + ',' + str(num_qubits) + '): ' + str(i))
        G = qsee.gradient.qng(qc.copy(), thetas, qsee.ansatz.create_Wchain_layered_ansatz, num_layers = num_layers)
        grad_loss = qsee.measure.grad_loss(
            qc, 
            qsee.ansatz.create_Wchain_layered_ansatz,
            thetas, num_layers = num_layers)

        grad = np.linalg.inv(G) @ grad_loss
        grads.append(grad)
    t = []
    for grad in grads:
        t.append(grad[-1])
    print(np.var(t))
    ts.append(np.var(t))
np.savetxt("./barren_wchain6_8.csv", ts, delimiter=",")
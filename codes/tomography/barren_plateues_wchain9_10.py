import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.fubini_study
layers = range(9, 11)
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
        G = qtm.fubini_study.qng(qc.copy(), thetas, qtm.ansatz.create_Wchain_layered_ansatz, num_layers = num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.ansatz.create_Wchain_layered_ansatz,
            thetas, num_layers = num_layers)

        grad = np.linalg.inv(G) @ grad_loss
        grads.append(grad)
    t = []
    for grad in grads:
        t.append(grad[-1])
    print(np.var(t))
    ts.append(np.var(t))
np.savetxt("./barren_wchain9_10.csv", ts, delimiter=",")
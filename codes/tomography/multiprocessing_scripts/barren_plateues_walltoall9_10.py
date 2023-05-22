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
    n_walltoall = qtm.ansatz.calculate_n_walltoall(num_qubits)
    thetas = np.ones(num_layers* 3 * num_qubits + num_layers*n_walltoall)

    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi)
    loss_values = []
    thetass = []
    for i in range(0, 200):
        if i % 20 == 0:
            print('W_alltoall: (' + str(num_layers) + ',' + str(num_qubits) + '): ' + str(i))
        G = qtm.fubini_study.qng(qc.copy(), thetas, qtm.ansatz.create_Walltoall_layered_ansatz, num_layers = num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.ansatz.create_Walltoall_layered_ansatz,
            thetas, num_layers = num_layers)

        grad = np.linalg.inv(G) @ grad_loss
        grads.append(grad)
    t = []
    for grad in grads:
        t.append(grad[-1])
    print(np.var(t))
    ts.append(np.var(t))
np.savetxt("./barren_walltoall9_10.csv", ts, delimiter=",")
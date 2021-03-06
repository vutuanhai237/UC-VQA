import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.encoding


def run_w(num_layers, num_qubits):
    thetas = np.ones(num_qubits*num_layers*5)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    loss_values_w = []
    thetass_w = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W (' + str(num_layers) + ' layer): ', i)
        # G = qtm.fubini_study.calculate_linear_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.ansatz.create_Wchecker_linear, 
            thetas, num_layers = num_layers)
        # grad1 = np.real(np.linalg.inv(G) @ grad_loss)
        thetas -= qtm.constant.learning_rate*grad_loss    
        qc_copy = qtm.ansatz.create_Wchecker_linear(qc.copy(), thetas, num_layers)
        loss = qtm.loss.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_w.append(loss)
        thetass_w.append(thetas.copy())

    traces_w, fidelities_w = [], []
    for thetas in thetass_w:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.ansatz.create_linear_state(qc, thetas, num_layers = num_layers)
        psi , rho_psi = qtm.base.extract_state(qc)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.ansatz.create_w_state(num_qubits)
psi_hat , rho_psi_hat = qtm.base.extract_state(qc1)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces_w.append(trace)
        fidelities_w.append(fidelity)
    print('Writting ...')
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/loss_values_w.csv", loss_values_w, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/thetass_w.csv", thetass_w, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/traces_w.csv", traces_w, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/fidelities_w.csv", fidelities_w, delimiter=",")



if __name__ == "__main__":
    # creating thread
    num_qubits = 5
    num_layers = [1, 2, 3, 4, 5]
   
    t_w = []

    for i in num_layers:
        t_w.append(multiprocessing.Process(target = run_w, args=(i, num_qubits)))

    for i in range(0, len(num_layers)):
        t_w[i].start()

    for i in range(0, len(num_layers)):
        t_w[i].join()

    print("Done!")
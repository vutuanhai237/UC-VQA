import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding


def run_w(num_layers, num_qubits):
    thetas = np.ones(num_qubits*num_layers*5)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    loss_values_w = []
    thetass_w = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W (' + str(num_layers) + ' layer): ', i)
        G = qtm.fubini_study.calculate_koczor_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_Wchecker_koczor, 
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        grad1 = np.real(np.linalg.inv(G) @ grad_loss)
        if i == 0:
            m, v = list(np.zeros(thetas.shape[0])), list(np.zeros(thetas.shape[0]))
        thetas = qtm.base.adam(thetas, m, v, i, grad1)    
        qc_copy = qtm.nqubit.create_Wchecker_koczor(qc.copy(), thetas, num_layers)
        loss = qtm.base.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_w.append(loss)
        thetass_w.append(thetas)

    traces_w, fidelities_w = [], []
    for thetas in thetass_w:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.nqubit.create_koczor_state(qc, thetas, num_layers = num_layers)
        psi = qiskit.quantum_info.Statevector.from_instruction(qc)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.nqubit.create_w_state(qc1)
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc1)
        rho_psi_hat = qiskit.quantum_info.DensityMatrix(psi_hat)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces_w.append(trace)
        fidelities_w.append(fidelity)
    print('Writting ...')
    np.savetxt("../../experiments/koczor_ansatz_15layer_qngadam/" + str(num_layers) + "/loss_values_w.csv", loss_values_w, delimiter=",")
    np.savetxt("../../experiments/koczor_ansatz_15layer_qngadam/" + str(num_layers) + "/thetass_w.csv", thetass_w, delimiter=",")
    np.savetxt("../../experiments/koczor_ansatz_15layer_qngadam/" + str(num_layers) + "/traces_w.csv", traces_w, delimiter=",")
    np.savetxt("../../experiments/koczor_ansatz_15layer_qngadam/" + str(num_layers) + "/fidelities_w.csv", fidelities_w, delimiter=",")



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
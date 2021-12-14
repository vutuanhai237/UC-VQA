import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base_qtm, qtm.constant, qtm.qtm_nqubit, qtm.fubini_study, qtm.encoding
import importlib
importlib.reload(qtm.base_qtm)
importlib.reload(qtm.constant)
importlib.reload(qtm.qtm_1qubit)
importlib.reload(qtm.qtm_nqubit)
# Init parameters

# For arbitrary initial state

theta = np.pi/3

def run_w(num_layers, num_qubits):
    # W

    thetas = np.random.random((num_qubits*5 - 4)*num_layers)

    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    loss_values_w = []
    thetass_w = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W (' + str(num_layers) + ' layer): ', i)
        G = qtm.fubini_study.calculate_alternative_layered_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base_qtm.grad_loss(
            qc, 
            qtm.qtm_nqubit.create_Wchecker_alternating_layered, 
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.inv(G) @ grad_loss))   
        qc_copy = qtm.qtm_nqubit.create_Wchecker_alternating_layered(qc.copy(), thetas, num_layers)
        loss = qtm.base_qtm.loss_basis(qtm.base_qtm.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_w.append(loss)
        thetass_w.append(thetas)

    traces_w, fidelities_w = [], []
    for thetas in thetass_w:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.qtm_nqubit.create_alternating_layerd_state(qc, thetas, num_layers = num_layers)
        psi = qiskit.quantum_info.Statevector.from_instruction(qc)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.qtm_nqubit.create_w_state(qc1)
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc1)
        rho_psi_hat = qiskit.quantum_info.DensityMatrix(psi_hat)
        # Calculate the metrics
        trace, fidelity = qtm.base_qtm.get_metrics(psi, psi_hat)
        traces_w.append(trace)
        fidelities_w.append(fidelity)
    print('Writting ... ' + str(num_qubits))
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/loss_values_w.csv", loss_values_w, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/thetass_w.csv", thetass_w, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/traces_w.csv", traces_w, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/fidelities_w.csv", fidelities_w, delimiter=",")




if __name__ == "__main__":
    # creating thread
    num_qubits = [3, 4, 5, 6, 7, 8, 9, 10]
    num_layers = 1
   
    t_w = []

    for i in num_qubits:
        t_w.append(multiprocessing.Process(target = run_w, args=(num_layers, i)))

    for i in range(0, len(num_qubits)):
        t_w[i].start()

    for i in range(0, len(num_qubits)):
        t_w[i].join()

    print("Done!")
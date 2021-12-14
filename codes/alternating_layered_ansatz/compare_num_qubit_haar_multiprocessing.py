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

def run_haar(num_layers, num_qubits):

    thetas = np.random.random((num_qubits*5 - 4)*num_layers)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')

    loss_values_haar = []
    thetass_haar = []
    for i in range(0, 10):
        if i % 20 == 0:
            print('Haar (' + str(num_layers) + ' layer): ', i)
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        G = qtm.fubini_study.calculate_alternative_layered_state(qc.copy(), thetas, num_layers)
        qc = encoder.qcircuit
        grad_loss = qtm.base_qtm.grad_loss(
            qc, 
            qtm.qtm_nqubit.create_haarchecker_alternating_layered, 
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers, encoder = encoder)

        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.inv(G) @ grad_loss))   
        qc_copy = qtm.qtm_nqubit.create_haarchecker_alternating_layered(qc.copy(), thetas, num_layers, encoder)
        loss = qtm.base_qtm.loss_basis(qtm.base_qtm.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_haar.append(loss)
        thetass_haar.append(thetas)

    traces_haar, fidelities_haar = [], []
    for thetas in thetass_haar:
    # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.qtm_nqubit.create_alternating_layerd_state(qc, thetas, num_layers = num_layers)
        psi = qiskit.quantum_info.Statevector.from_instruction(qc)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        # Get |psi~> = U_target|000...>
        qc1 = encoder.qcircuit
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc1)
        rho_psi_hat = qiskit.quantum_info.DensityMatrix(psi_hat)
        # Calculate the metrics
        trace, fidelity = qtm.base_qtm.get_metrics(psi, psi_hat)
        traces_haar.append(trace)
        fidelities_haar.append(fidelity)
    print('Writting ... ' + str(num_qubits))
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/loss_values_haar.csv", loss_values_haar, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/thetass_haar.csv", thetass_haar, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/traces_haar.csv", traces_haar, delimiter=",")
    np.savetxt("../../experiments/alternating_layered_ansatz/" + str(num_qubits) + "/fidelities_haar.csv", fidelities_haar, delimiter=",")




if __name__ == "__main__":
    # creating thread
    num_qubits = [3, 4, 5, 6, 7, 8, 9, 10]
    num_layers = 1
   
    t_haar = []

    for i in num_qubits:
        t_haar.append(multiprocessing.Process(target = run_haar, args=(num_layers, i)))

    for i in range(0, len(num_qubits)):
        t_haar[i].start()

    for i in range(0, len(num_qubits)):
        t_haar[i].join()

    print("Done!")
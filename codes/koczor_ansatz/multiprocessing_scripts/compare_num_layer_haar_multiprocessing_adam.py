import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding


def run_haar(num_layers, num_qubits):
 
    psi = 2*np.random.rand(2**num_qubits)-1
    # Haar
    thetas = np.ones(num_qubits*num_layers*5)

    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')

    loss_values_haar = []
    thetass_haar = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('Haar (' + str(num_layers) + ' layer): ', i)
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        # G = qtm.fubini_study.calculate_linear_state(qc.copy(), thetas, num_layers)
        qc = encoder.qcircuit
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_haarchecker_linear, 
            thetas, num_layers = num_layers, encoder = encoder)
        # grad1 = np.real(np.linalg.inv(G) @ grad_loss)
        if i == 0:
            m, v = list(np.zeros(thetas.shape[0])), list(np.zeros(thetas.shape[0]))
        thetas = qtm.base.adam(thetas, m, v, i, grad_loss)    
        qc_copy = qtm.nqubit.create_haarchecker_linear(qc.copy(), thetas, num_layers, encoder)
        loss = qtm.base.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_haar.append(loss)
        thetass_haar.append(thetas)

    traces_haar, fidelities_haar = [], []
    for thetas in thetass_haar:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.nqubit.create_linear_state(qc, thetas, num_layers = num_layers)
        psi , rho_psi = qtm.base.extract_state(qc)
        # Get |psi~> = U_target|000...>
        qc1 = encoder.qcircuit
psi_hat , rho_psi_hat = qtm.base.extract_state(qc1)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces_haar.append(trace)
        fidelities_haar.append(fidelity)
    print('Writting ... ' + str(num_layers))
    np.savetxt("../../experiments/linear_ansatz_15layer_adam/" + str(num_layers) + "/loss_values_haar.csv", loss_values_haar, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_adam/" + str(num_layers) + "/thetass_haar.csv", thetass_haar, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_adam/" + str(num_layers) + "/traces_haar.csv", traces_haar, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_adam/" + str(num_layers) + "/fidelities_haar.csv", fidelities_haar, delimiter=",")



if __name__ == "__main__":
    # creating thread
    num_qubits = 5
    num_layers = [1, 2, 3, 4, 5]
   
    t_haar = []

    for i in num_layers:
        t_haar.append(multiprocessing.Process(target = run_haar, args=(i, num_qubits)))

    for i in range(0, len(num_layers)):
        t_haar[i].start()

    for i in range(0, len(num_layers)):
        t_haar[i].join()

    print("Done!")
import qiskit
import numpy as np
import sys

sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding
import importlib
import multiprocessing

importlib.reload(qtm.base)
importlib.reload(qtm.constant)
importlib.reload(qtm.onequbit)
importlib.reload(qtm.nqubit)
importlib.reload(qtm.fubini_study)


def run_walternating(num_layers, num_qubits):
    n_alternating = 0
    for i in range(0, num_layers):
        n_alternating += qtm.nqubit.calculate_n_walternating(i, num_qubits)
    thetas = np.ones(n_alternating + 3 * num_layers * num_qubits)

    psi = 2 * np.random.rand(2**num_qubits) - 1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')


    loss_values = []
    thetass = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W_alternating: (' + str(num_layers) + ',' +
                  str(num_qubits) + '): ' + str(i))
        qc = encoder.qcircuit
        G = qtm.fubini_study.calculate_Walternating_state(qc.copy(), thetas, num_layers)
        
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_Walternatingchecker_haar,
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.pinv(G) @ grad_loss)) 
        thetass.append(thetas.copy())
        qc_copy = qtm.nqubit.create_Walternatingchecker_haar(qc.copy(), thetas, num_layers)  
        loss = qtm.base.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)

    traces = []
    fidelities = []

    for thetas in thetass:
        # Get |psi> = U_gen|000...>
        qc1 = encoder.qcircuit
        psi = qiskit.quantum_info.Statevector.from_instruction(qc1)
        # Get |psi~> = U_target|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.nqubit.create_Walternating_layerd_state(
            qc, thetas, num_layers=num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)

    print('Writting ... ' + str(num_layers) + ' layers,' + str(num_qubits) +
          ' qubits')

    np.savetxt("../../experiments/tomography_walternating_" + str(num_layers) +
               "/" + str(num_qubits) + "/loss_values_qng.csv",
               loss_values,
               delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_" + str(num_layers) +
               "/" + str(num_qubits) + "/thetass_qng.csv",
               thetass,
               delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_" + str(num_layers) +
               "/" + str(num_qubits) + "/traces_qng.csv",
               traces,
               delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_" + str(num_layers) +
               "/" + str(num_qubits) + "/fidelities_qng.csv",
               fidelities,
               delimiter=",")


if __name__ == "__main__":
    # creating thread

    num_layers = [1, 2, 3, 4, 5]
    num_qubits = [3, 4, 5]
    t_walternatings = []

    for i in num_layers:
        for j in num_qubits:
            t_walternatings.append(
                multiprocessing.Process(target=run_walternating, args=(i, j)))

    for t_walternating in t_walternatings:
        t_walternating.start()

    for t_walternating in t_walternatings:
        t_walternating.join()

    print("Done!")
import qiskit
import numpy as np
import sys
import qtm.base
import qtm.constant
import qtm.ansatz
import qtm.fubini_study
import qtm.encoding
import multiprocessing


def run_walltoall(num_layers, num_qubits):
    thetas = np.ones((num_qubits*num_layers*5))
    theta = np.pi / 2
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    loss_values = []
    thetass = []
    for i in range(0, 100):
        if i % 20 == 0:
            print('Linear ansatz GHZ: (' + str(num_layers) +
                  ',' + str(num_qubits) + '): ' + str(i))

        G = qtm.fubini_study.qng(
            qc.copy(), thetas, qtm.ansatz.create_GHZchecker_linear, num_layers)
        grad_loss = qtm.base.grad_loss(
            qc,
            qtm.ansatz.create_GHZchecker_linear,
            thetas, r=1/2, s=np.pi/2, num_layers=num_layers, theta=theta)
        thetas = np.real(thetas - qtm.constant.learning_rate *
                         (np.linalg.inv(G) @ grad_loss))
        qc_copy = qtm.ansatz.create_GHZchecker_linear(
            qc.copy(), thetas, num_layers, theta)
        loss = qtm.loss.loss_fubini_study(qtm.base.measure(
            qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)
        thetass.append(thetas)
    np.savetxt("../experiments/linear_ansatz_ghz/" + str(num_qubits) +
               "/loss_values_qng.csv", loss_values, delimiter=",")
    np.savetxt("../experiments/linear_ansatz_ghz/" +
               str(num_qubits) + "/thetass_qng.csv", thetass, delimiter=",")
    traces = []
    fidelities = []
    i = 0
    for thetas in thetass:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.ansatz.create_linear_state(qc, thetas, num_layers)
        psi , rho_psi = qtm.base.extract_state(qc)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.ansatz.create_ghz_state(num_qubits, theta)
        psi_hat , rho_psi_hat = qtm.base.extract_state(qc1)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    np.savetxt("../experiments/linear_ansatz_ghz/" +
               str(num_qubits) + "/traces_qng.csv", traces, delimiter=",")
    np.savetxt("../experiments/linear_ansatz_ghz/" + str(num_qubits) +
               "/fidelities_qng.csv", fidelities, delimiter=",")


if __name__ == "__main__":
    # creating thread

    num_qubits = [3, 4, 5, 6, 7, 8, 9, 10]
    linear_ansatzs = []
    i = 2
    for j in num_qubits:
        linear_ansatzs.append(multiprocessing.Process(
            target=run_walltoall, args=(i, j)))

    for linear_ansatz in linear_ansatzs:
        linear_ansatz.start()

    for linear_ansatz in linear_ansatzs:
        linear_ansatz.join()

    print("Done!")

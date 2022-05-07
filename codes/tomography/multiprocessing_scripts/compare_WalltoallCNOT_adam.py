
import multiprocessing
import importlib
import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.encoding
import qtm.fubini_study
import qtm.nqubit
import qtm.constant
import qtm.base


def run_walltoall(num_layers, num_qubits):

    thetas = np.ones(num_layers * 3 * num_qubits)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(0, num_qubits))

    loss_values = []
    thetass = []

    for i in range(0, 400):
        if i % 20 == 0:
            print('W_alltoall: (' + str(num_layers) +
                  ',' + str(num_qubits) + '): ' + str(i))

        grad_loss = qtm.base.grad_loss(
            qc,
            qtm.nqubit.create_WalternatingCNOT_layerd_state,
            thetas, r=1/2, s=np.pi/2, num_layers=num_layers)
        if i == 0:
            m, v = list(np.zeros(thetas.shape[0])), list(
                np.zeros(thetas.shape[0]))
        thetas = qtm.optimizer.adam(thetas, m, v, i, grad_loss)
        thetass.append(thetas.copy())
        qc_copy = qtm.nqubit.create_WalternatingCNOT_layerd_state(
            qc.copy(), thetas, num_layers)
        loss = qtm.loss.loss_basis(qtm.base.measure(
            qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)

    traces = []
    fidelities = []

    for thetas in thetass:
        # Get |psi~> = U_target|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.nqubit.create_WalternatingCNOT_layerd_state(
            qc, thetas, num_layers=num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    print('Writting ... ' + str(num_layers) +
          ' layers,' + str(num_qubits) + ' qubits')

    np.savetxt("../../experiments/tomographyCNOT/tomography_walltoall_" + str(num_layers) +
               "/" + str(num_qubits) + "/loss_values_adam.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomographyCNOT/tomography_walltoall_" + str(num_layers) +
               "/" + str(num_qubits) + "/thetass_adam.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomographyCNOT/tomography_walltoall_" +
               str(num_layers) + "/" + str(num_qubits) + "/traces_adam.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomographyCNOT/tomography_walltoall_" + str(num_layers) +
               "/" + str(num_qubits) + "/fidelities_adam.csv", fidelities, delimiter=",")


if __name__ == "__main__":
    # creating thread

    num_layers = [1, 2, 3, 4, 5]
    num_qubits = [3, 4, 5]
    t_walltoalls = []

    for i in num_layers:
        for j in num_qubits:
            t_walltoalls.append(multiprocessing.Process(
                target=run_walltoall, args=(i, j)))

    for t_walltoall in t_walltoalls:
        t_walltoall.start()

    for t_walltoall in t_walltoalls:
        t_walltoall.join()

    print("Done!")

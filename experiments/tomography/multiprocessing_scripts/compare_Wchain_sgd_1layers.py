import qiskit
import numpy as np
import sys
sys.path.insert(1, '../../')
import qsee.measure, qsee.backend.constant, qsee.ansatz, qsee.gradient, qsee.state

import multiprocessing

def run_wchain(num_layers, num_qubits):

    thetas = np.ones(num_layers*num_qubits*4)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(0, num_qubits))

    loss_values = []
    thetass = []

    for i in range(0, 400):
        if i % 20 == 0:
            print('W_chain: (' + str(num_layers) + ',' + str(num_qubits) + '): ' + str(i))

        grad_loss = qsee.measure.grad_loss(
            qc, 
            qsee.ansatz.create_Wchain_layerd_state,
            thetas, num_layers = num_layers)
        thetas -= qsee.constant.learning_rate*(grad_loss) 
        thetass.append(thetas.copy())
        qc_copy = qsee.ansatz.create_Wchain_layerd_state(qc.copy(), thetas, num_layers)  
        loss = qsee.loss.loss_basis(qsee.measure.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)

    traces = []
    fidelities = []

    for thetas in thetass:
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qsee.ansatz.create_Wchain_layerd_state(qc, thetas, num_layers = num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qsee.measure.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    print('Writting ... ' + str(num_layers) + ' layers,' + str(num_qubits) + ' qubits')

    np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + "/" + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + "/" + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + "/" + str(num_qubits) + "/traces.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + "/" + str(num_qubits) + "/fidelities.csv", fidelities, delimiter=",")

if __name__ == "__main__":
    # creating thread
    
    num_layers = [1]
    num_qubits = [2, 3, 4, 5, 6]
    t_wchains = []

    for i in num_layers:
        for j in num_qubits:
            t_wchains.append(multiprocessing.Process(target = run_wchain, args=(i, j)))

    for t_wchain in t_wchains:
        t_wchain.start()

    for t_wchain in t_wchains:
        t_wchain.join()

    print("Done!")
import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding
import multiprocessing

def run_walltoall(num_layers, num_qubits):
    n_walltoall = qtm.nqubit.calculate_n_walltoall(num_qubits)
    thetas = np.ones(num_layers* 3 * num_qubits + num_layers*n_walltoall)

    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')


    loss_values = []
    thetass = []
    for i in range(0, 400):
        if i % 20 == 0:
            print('W_alltoall: (' + str(num_layers) + ',' + str(num_qubits) + '): ' + str(i))
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        G = qtm.fubini_study.calculate_Walltoall_state(qc, thetas, num_layers)
        qc = encoder.qcircuit
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_Walltoallchecker_haar,
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas -= qtm.constant.learning_rate*(grad_loss) 
        thetass.append(thetas.copy())
        qc_copy = qtm.nqubit.create_Walltoallchecker_haar(qc.copy(), thetas, num_layers)  
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
        qc = qtm.nqubit.create_Walltoall_layerd_state(qc, thetas, num_layers = num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)

    print('Writting ... ' + str(num_layers) + ' layers,' + str(num_qubits) + ' qubits')

    np.savetxt("../../experiments/tomography_walltoall_" + str(num_layers) + "/" + str(num_qubits) + "/loss_values_qng.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_" + str(num_layers) + "/" + str(num_qubits) + "/thetass_qng.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_" + str(num_layers) + "/" + str(num_qubits) + "/traces_qng.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_" + str(num_layers) + "/" + str(num_qubits) + "/fidelities_qng.csv", fidelities, delimiter=",")

if __name__ == "__main__":
    # creating thread
    
    num_layers = [1, 2, 3, 4, 5]
    num_qubits = [3,4,5]
    t_walltoalls = []

    for i in num_layers:
        for j in num_qubits:
            t_walltoalls.append(multiprocessing.Process(target = run_walltoall, args=(i, j)))

    for t_walltoall in t_walltoalls:
        t_walltoall.start()

    for t_walltoall in t_walltoalls:
        t_walltoall.join()

    print("Done!")
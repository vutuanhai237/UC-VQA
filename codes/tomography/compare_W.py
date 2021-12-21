from math import tau
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

def w_chain():
    num_qubits = 3
    num_layers = 1
    thetas = np.ones(num_layers*num_qubits*4)
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')


    loss_values = []
    thetass = []

    # qc = encoder.qcircuit
    # qc = qtm.nqubit.create_Wchainchecker_haar(qc, thetas, num_layers)

    for i in range(0, 2):
        if i % 20 == 0:
            print('W_chain: ', i)
        qc = encoder.qcircuit
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_Wchainchecker_haar,
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas -= qtm.constant.learning_rate*(grad_loss) 
        thetass.append(thetas.copy())
        qc_copy = qtm.nqubit.create_Wchainchecker_haar(qc.copy(), thetas, num_layers)  
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
        qc = qtm.nqubit.create_Wchain_layerd_state(qc, thetas, num_layers = num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    print('Writting ... ' + str(num_qubits))

    np.savetxt("../../experiments/tomography_wchain_/" + str(num_layers) + '/' + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomography_wchain_/" + str(num_layers) + '/' + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomography_wchain_/" + str(num_layers) + '/' + str(num_qubits) + "/traces.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomography_wchain_/" + str(num_layers) + '/' + str(num_qubits) + "/fidelities.csv", fidelities, delimiter=",")

def w_alternating():
    num_qubits = 3
    num_layers = 1
    n_alternating = 0
    for i in range(0, num_layers + 1):
        n_alternating += qtm.nqubit.calculate_n_walternating(i, num_qubits)
    thetas = np.ones(n_alternating + 3 * num_layers * num_qubits)

    for i in range(0, len(thetas)):
        thetas[i] += i

    psi = 2 * np.random.rand(2**num_qubits) - 1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')


    loss_values = []
    thetass = []
    for i in range(0, 2):
        if i % 20 == 0:
            print('W_alternating: ', i)
        qc = encoder.qcircuit
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.nqubit.create_Walternatingchecker_haar,
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas -= qtm.constant.learning_rate*(grad_loss) 
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
        qc = qtm.nqubit.create_Walternating_layerd_state(qc, thetas, num_layers = num_layers).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    print('Writting ... ' + str(num_qubits))

    np.savetxt("../../experiments/tomography_walternating_/" + str(num_layers) + '/' + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_/" + str(num_layers) + '/' + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_/" + str(num_layers) + '/' + str(num_qubits) + "/traces.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomography_walternating_/" + str(num_layers) + '/' + str(num_qubits) + "/fidelities.csv", fidelities, delimiter=",")

def w_alltoall():
    num_qubits = 3
    num_layers = 1
    n_walltoall = qtm.nqubit.calculate_n_walltoall(num_qubits)
    thetas = np.ones(num_layers* 3 * num_qubits + num_layers*n_walltoall)
    for i in range(0, len(thetas)):
        thetas[i] += i
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')


    loss_values = []
    thetass = []
    for i in range(0, 2):
        if i % 20 == 0:
            print('W_chain: ', i)
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
    print('Writting ... ' + str(num_qubits))


    np.savetxt("../../experiments/tomography_walltoall_/" + str(num_layers) + '/' + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_/" + str(num_layers) + '/' + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_/" + str(num_layers) + '/' + str(num_qubits) + "/traces.csv", traces, delimiter=",")
    np.savetxt("../../experiments/tomography_walltoall_/" + str(num_layers) + '/' + str(num_qubits) + "/fidelities.csv", fidelities, delimiter=",")

if __name__ == "__main__":
    # creating thread
    
    ts = []
    ts.append(multiprocessing.Process(target = w_chain))
    ts.append(multiprocessing.Process(target = w_alternating))
    ts.append(multiprocessing.Process(target = w_alltoall))
    for t in ts:
        t.start()

    for t in ts:
        t.join()

    print("Done!")
import qiskit
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../')
import qtm.base_qtm, qtm.constant, qtm.qtm_nqubit, qtm.fubini_study, qtm.encoding
import threading
# Init parameters


def run_ghz(num_qubits, iter):
    num_layers = 1
    thetas_origin = np.random.uniform(low = 0, high = 2*np.pi, size = num_qubits*num_layers*5)
    # For determine GHZ state
    theta = np.pi/ 3
    
    # GHZ
    thetas = thetas_origin.copy()
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)

    loss_values_ghz = []
    thetass_ghz = []
    print('GHZ')
    for i in range(0, iter):
        if i % 20 == 0:
            print('GHZ', i)
        # fubini_study for binho_state is same for koczor state
        G = qtm.fubini_study.calculate_koczor_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base_qtm.grad_loss(
            qc, 
            qtm.qtm_nqubit.create_GHZchecker_binho,
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers, theta = theta)
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.inv(G) @ grad_loss))   
        qc_copy = qtm.qtm_nqubit.create_GHZchecker_binho(qc.copy(), thetas, num_layers, theta)
        
        loss = qtm.base_qtm.loss_basis(qtm.base_qtm.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_ghz.append(loss)
        thetass_ghz.append(thetas)

    traces_ghz, fidelities_ghz = [], []
    for thetas in thetass_ghz:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.qtm_nqubit.create_binho_state(qc, thetas, num_layers = num_layers)
        psi = qiskit.quantum_info.Statevector.from_instruction(qc)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.qtm_nqubit.create_ghz_state(qc1, theta = theta)
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc1)
        rho_psi_hat = qiskit.quantum_info.DensityMatrix(psi_hat)
        # Calculate the metrics
        trace, fidelity = qtm.base_qtm.get_metrics(psi, psi_hat)
        traces_ghz.append(trace)
        fidelities_ghz.append(fidelity)

    np.savetxt("./loss_values_ghz.csv", loss_values_ghz, delimiter=",")
    np.savetxt("./thetass_ghz.csv", thetass_ghz, delimiter=",")
    np.savetxt("./traces_ghz.csv", traces_ghz, delimiter=",")
    np.savetxt("./fidelities_ghz.csv", fidelities_ghz, delimiter=",")

def run_w(num_qubits, iter):
    num_layers = 1
    thetas_origin = np.random.uniform(low = 0, high = 2*np.pi, size = num_qubits*num_layers*5)
    thetas = thetas_origin.copy()
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    print('W')
    loss_values_w = []
    thetass_w = []
    for i in range(0, iter):
        if i % 20 == 0:
            print('W', i)
        G = qtm.fubini_study.calculate_koczor_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base_qtm.grad_loss(
            qc, 
            qtm.qtm_nqubit.create_Wchecker_binho, 
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.inv(G) @ grad_loss))   
        qc_copy = qtm.qtm_nqubit.create_Wchecker_binho(qc.copy(), thetas, num_layers)
        loss = qtm.base_qtm.loss_basis(qtm.base_qtm.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_w.append(loss)
        thetass_w.append(thetas)

    import qtm.custom_gate
    traces_w, fidelities_w = [], []
    for thetas in thetass_w:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.qtm_nqubit.create_binho_state(qc, thetas, num_layers = num_layers)
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

    np.savetxt("./loss_values_w.csv", loss_values_w, delimiter=",")
    np.savetxt("./thetass_w.csv", thetass_w, delimiter=",")
    np.savetxt("./traces_w.csv", traces_w, delimiter=",")
    np.savetxt("./fidelities_w.csv", fidelities_w, delimiter=",")

def run_haar(num_qubits, iter):

    # For arbitrary initial state
    num_layers = 1
    thetas_origin = np.random.uniform(low = 0, high = 2*np.pi, size = num_qubits*num_layers*5)

    psi = 2*np.random.rand(2**num_qubits)-1

    # Haar
    print('Haar')
    thetas = thetas_origin.copy()

    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')

    loss_values_haar = []
    thetass_haar = []
    for i in range(0, iter):
        if i % 20 == 0:
            print('Haar', i)
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        G = qtm.fubini_study.calculate_koczor_state(qc.copy(), thetas, num_layers)
        qc = encoder.qcircuit
        grad_loss = qtm.base_qtm.grad_loss(
            qc, 
            qtm.qtm_nqubit.create_haarchecker_binho, 
            thetas, r = 1/2, s = np.pi/2, num_layers = num_layers, encoder = encoder)
        
        thetas = np.real(thetas - qtm.constant.learning_rate*(np.linalg.inv(G) @ grad_loss))   
        qc_copy = qtm.qtm_nqubit.create_haarchecker_binho(qc.copy(), thetas, num_layers, encoder)
        loss = qtm.base_qtm.loss_basis(qtm.base_qtm.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_haar.append(loss)
        thetass_haar.append(thetas)

    traces_haar, fidelities_haar = [], []
    for thetas in thetass_haar:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.qtm_nqubit.create_binho_state(qc, thetas, num_layers = num_layers)
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

    np.savetxt("./loss_values_haar.csv", loss_values_haar, delimiter=",")
    np.savetxt("./thetass_haar.csv", thetass_haar, delimiter=",")
    np.savetxt("./traces_haar.csv", traces_haar, delimiter=",")
    np.savetxt("./fidelities_haar.csv", fidelities_haar, delimiter=",")

if __name__ == "__main__":
    # creating thread
    qubits = [3,4,5,6,7,8,9,10]
    qubits_haar = [3,4,5]

    t_ghz = []
    t_w = []
    t_haar = []
    for i in qubits:
        t_ghz.append(threading.Thread(target = run_ghz, args=(i, 300)))
        t_w.append(threading.Thread(target = run_w, args=(i, 300)))
    for i in qubits_haar:
        t_haar.append(threading.Thread(target = run_haar, args=(i, 300)))

    for i in range(0, len(qubits)):
        t_ghz[i].start()
        t_w[i].start()
    for i in range(0, len(qubits_haar)):
        t_haar[i].start()
    for i in range(0, len(qubits)):
        t_ghz[i].join()
        t_w[i].join()
    for i in range(0, len(qubits_haar)):
        t_haar[i].join()

    print("Done!")

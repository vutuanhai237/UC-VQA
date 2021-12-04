import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base_qtm, qtm.constant, qtm.qtm_nqubit, qtm.fubini_study, qtm.encoding


def run_ghz(num_layers, num_qubits):
    # GHZ
    
    thetas_origin = np.random.uniform(low = 0, high = 2*np.pi, size = num_qubits*num_layers*5)
    # For determine GHZ state
    theta = np.random.uniform(0, 2*np.pi)
    thetas = thetas_origin.copy()
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)

    loss_values_ghz = []
    thetass_ghz = []
    for i in range(0, 200):
        # fubini_study for binho_state is same for koczor state
        if i % 20 == 0:
            print('GHZ (' + str(num_layers) + ' layer): ', i)
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
        # Plot loss value in 100 steps
    print('Writting ...')

    np.savetxt("./" + str(num_layers) + "/loss_values_ghz.csv", loss_values_ghz, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/thetass_ghz.csv", thetass_ghz, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/traces_ghz.csv", traces_ghz, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/fidelities_ghz.csv", fidelities_ghz, delimiter=",")
def run_w(num_layers, num_qubits):
    
    thetas_origin = np.random.uniform(low = 0, high = 2*np.pi, size = num_qubits*num_layers*5)
    thetas = thetas_origin.copy()
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    loss_values_w = []
    thetass_w = []
    for i in range(0, 200):
        if i % 20 == 0:
            print('W (' + str(num_layers) + ' layer): ', i)
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
    print('Writting ...')
    np.savetxt("./" + str(num_layers) + "/loss_values_w.csv", loss_values_w, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/thetass_w.csv", thetass_w, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/traces_w.csv", traces_w, delimiter=",")
    np.savetxt("./" + str(num_layers) + "/fidelities_w.csv", fidelities_w, delimiter=",")


if __name__ == "__main__":

    num_qubits = 3
    num_layers = [2, 3]
    t_ghz = []
    t_w = []

    for i in num_layers:
        t_ghz.append(multiprocessing.Process(target = run_ghz, args=(i, num_qubits)))
        t_w.append(multiprocessing.Process(target = run_w, args=(i, num_qubits)))

    for i in range(0, len(num_layers)):
        t_ghz[i].start()
        t_w[i].start()

    for i in range(0, len(num_layers)):
        t_ghz[i].join()
        t_w[i].join()

    print("Done!")
import qiskit
import numpy as np
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.encoding
import importlib
importlib.reload(qtm.base)
importlib.reload(qtm.constant)

importlib.reload(qtm.ansatz)
# Init parameters

# For arbitrary initial state

def run_ghz(num_layers, num_qubits):
    # GHZ
  
    theta = np.pi/3
    thetas = np.ones(num_qubits*num_layers*5)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)

    loss_values_ghz = []
    thetass_ghz = []
    for i in range(0, 400):
        # fubini_study for linear_state is same for linear state
        if i % 20 == 0:
            print('GHZ (' + str(num_layers) + ' layer): ', i)
        # G = qtm.fubini_study.calculate_linear_state(qc.copy(), thetas, num_layers)
        grad_loss = qtm.base.grad_loss(
            qc, 
            qtm.ansatz.create_GHZchecker_linear,
            thetas, num_layers = num_layers, theta = theta)
        # grad1 = np.real(np.linalg.inv(G) @ grad_loss)
        
        thetas -= qtm.constant.learning_rate*grad_loss
        qc_copy = qtm.ansatz.create_GHZchecker_linear(qc.copy(), thetas, num_layers, theta)  
        loss = qtm.loss.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values_ghz.append(loss)
        thetass_ghz.append(thetas.copy())
    traces_ghz, fidelities_ghz = [], []
    for thetas in thetass_ghz:
        # Get |psi> = U_gen|000...>
        qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc = qtm.ansatz.create_linear_state(qc, thetas, num_layers = num_layers)
        psi , rho_psi = qtm.base.extract_state(qc)
        # Get |psi~> = U_target|000...>
        qc1 = qiskit.QuantumCircuit(num_qubits, num_qubits)
        qc1 = qtm.ansatz.create_ghz_state(num_qubits, theta = theta)
psi_hat , rho_psi_hat = qtm.base.extract_state(qc1)
        # Calculate the metrics
        trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
        traces_ghz.append(trace)
        fidelities_ghz.append(fidelity)
        # Plot loss value in 100 steps
    print('Writting ...')

    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/loss_values_ghz.csv", loss_values_ghz, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/thetass_ghz.csv", thetass_ghz, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/traces_ghz.csv", traces_ghz, delimiter=",")
    np.savetxt("../../experiments/linear_ansatz_15layer_sgd/" + str(num_layers) + "/fidelities_ghz.csv", fidelities_ghz, delimiter=",")


if __name__ == "__main__":
    # creating thread
    num_qubits = 5
    num_layers = [1,2,3,4,5]
   
    t_ghz = []

    for i in num_layers:
        t_ghz.append(multiprocessing.Process(target = run_ghz, args=(i, num_qubits)))

    for i in range(0, len(num_layers)):
        t_ghz[i].start()

    for i in range(0, len(num_layers)):
        t_ghz[i].join()

    print("Done!")
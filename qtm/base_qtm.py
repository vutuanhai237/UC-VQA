from types import FunctionType
import numpy as np
import qiskit
import qtm.progress_bar
import scipy
import qtm.constant


def measure(qc: qiskit.QuantumCircuit, qubits):
    """Measuring the quantu circuit which fully measurement gates
    
    Args:
        qc (QuantumCircuit): Measured circuit
        qubits (Numpy array): List of measured qubit

    Returns:
        Float: Frequency of 00.. counter
    """
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], qubits[i])
    qobj = qiskit.assemble(qc, shots = qtm.constant.shots)  
    counts = (qiskit.Aer.get_backend('qasm_simulator')).run(qobj).result().get_counts()
    return counts.get("0" * qc.num_qubits, 0) / qtm.constant.shots

def trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        rho (DensityMatrix): first density matrix
        sigma (DensityMatrix): second density matrix
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1/2*sum(abs(w))

def trace_fidelity(rho, sigma):
    """ Calculating the fidelity metric

    Args:
        rho (DensityMatrix): first density matrix
        sigma (DensityMatrix): second density matrix
    """
    rho = rho.data
    sigma = sigma.data
    return np.trace(scipy.linalg.sqrtm((scipy.linalg.sqrtm(rho)).dot(rho)).dot(scipy.linalg.sqrtm(sigma)))

def get_metrics(psi, psi_hat):
    """
    Args:
        psi (Statevector): first state vector
        psi_hat (Statevector): second state vector
    
    Returns:
        Tuple: trace and fidelity
    """
    rho = qiskit.quantum_info.DensityMatrix(psi)
    sigma = qiskit.quantum_info.DensityMatrix(psi_hat)
    return qtm.base_qtm.trace_distance(rho, sigma), qtm.base_qtm.trace_fidelity(rho, sigma)

def grad_l(qc: qiskit.QuantumCircuit, create_circuit_func: FunctionType, thetas, r: float, s: float):
    """Return the gradient of the loss function

    L = 1 - |<psi~|psi>|^2 = 1 - P_0

    => nabla_L = - nabla_P_0 = - r (P_0(+s) - P_0(-s))

    Args:
        qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        create_circuit_func (Function): The creating circuit function
        thetas (Numpy array): Parameters
        r (float): r in parameter shift rule
        s (float): s in parameter shift rule

    Returns:
        Numpy array: The vector of gradient
    """
    gradient_l = np.zeros(len(thetas))
    for i in range(0, len(thetas)):
        thetas1, thetas2 = thetas.copy(), thetas.copy()
        thetas1[i] += s
        thetas2[i] -= s
        qc1 = create_circuit_func(qc.copy(), thetas1)
        qc2 = create_circuit_func(qc.copy(), thetas2)
        gradient_l[i] = -r*(
            qtm.base_qtm.measure(qc1, range(qc1.num_qubits)) - 
            qtm.base_qtm.measure(qc2, range(qc2.num_qubits))
        )
    return gradient_l
def loss_basis(measurement_value: float):
    """Return loss value for loss function L = 1 - P_0
    Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        measurement_value (Float): P_0 value

    Returns:
        Float: Loss value
    """
    return 1 - measurement_value

def fit(qc: qiskit.QuantumCircuit, num_steps: int, thetas, 
    create_circuit_func: FunctionType, 
    grad_func: FunctionType, 
    loss_func: FunctionType,
    verbose: int = 0):
    """Return the new thetas that fit with the circuit from create_circuit_func function

    Args:
        qc (QuantumCircuit): Fitting circuit
        num_steps (Int): number of iterations
        thetas (Numpy arrray): Parameters
        create_circuit_func (FunctionType): Added circuit function
        grad_func (FunctionType): Gradient function
        loss_func (FunctionType): Loss function
        verbose (Int): the seeing level of the fitting process (0: nothing, 1: progress bar, 2: one line per step)
    Returns:
        thetas (Numpy array): the optimized parameters
        loss_values (Numpy array): the list of loss_value
    """
    loss_values = []
    if verbose == 1:
        bar = qtm.progress_bar.ProgressBar(max_value = num_steps, disable = False)   
    for i in range(0, num_steps):
        thetas -= qtm.constant.learning_rate*grad_func(qc, create_circuit_func, thetas, 1/2, np.pi/2)
        qc_copy = create_circuit_func(qc.copy(), thetas)
        loss = loss_func(qtm.base_qtm.measure(qc_copy, range(qc_copy.num_qubits)))
        loss_values.append(loss)
        if verbose == 1:
            bar.update(1)
        if verbose == 2 and i % 10 == 0:
            print("Step " + str(i) + ": " + str(loss))
    if verbose == 1:      
        bar.close()
    return thetas, loss_values
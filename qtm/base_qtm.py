from types import FunctionType
import numpy as np
import qiskit, scipy
import qtm.progress_bar, qtm.constant, qtm.quantum_fisher



def measure(qc: qiskit.QuantumCircuit, qubits):
    """Measuring the quantu circuit which fully measurement gates
    
    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00.. cbit
    """
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], qubits[i])
    counts = qiskit.execute(qc, backend = qtm.constant.backend, shots = qtm.constant.num_shots).result().get_counts()
    return counts.get("0" * len(qubits), 0) / qtm.constant.num_shots

def trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1/2*sum(abs(w))

def trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix
    
    Returns:
        - float: trace metric has value from 0 to 1
    """
    rho = rho.data
    sigma = sigma.data
    return np.trace(scipy.linalg.sqrtm((scipy.linalg.sqrtm(rho)).dot(rho)).dot(scipy.linalg.sqrtm(sigma)))

def get_metrics(psi, psi_hat):
    """Get different metrics between the origin state and the reconstructed state
    
    Args:
        - psi (Statevector): first state vector
        - psi_hat (Statevector): second state vector
    
    Returns:
        - Tuple: trace and fidelity
    """
    rho = qiskit.quantum_info.DensityMatrix(psi)
    sigma = qiskit.quantum_info.DensityMatrix(psi_hat)
    return qtm.base_qtm.trace_distance(rho, sigma), qtm.base_qtm.trace_fidelity(rho, sigma)

def get_u_hat(thetas, create_circuit_func: FunctionType, num_qubits: int, **kwargs):
    """Return inverse of reconstructed gate

    Args:
        - thetas (Numpy array): Parameters
        - num_qubits (Int): number of qubit

    Returns:
        - Statevector: The state vector of when applying u_1q gate
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    if not kwargs:
        qc = create_circuit_func(qc, thetas).inverse()
    else:
        qc = create_circuit_func(qc, thetas, **kwargs).inverse()
    return qiskit.quantum_info.Statevector.from_instruction(qc)

def grad_loss(
    qc: qiskit.QuantumCircuit, 
    create_circuit_func: FunctionType, 
    thetas, r: float, s: float, **kwargs):
    """Return the gradient of the loss function
    
    L = 1 - |<psi~|psi>|^2 = 1 - P_0
    
    => nabla_L = - nabla_P_0 = - r (P_0(+s) - P_0(-s))

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - create_circuit_func (Function): The creating circuit function
        - thetas (Numpy array): Parameters
        - r (float): r in parameter shift rule
        - s (float): s in parameter shift rule
        - **kwargs: additional parameters for different create_circuit_func()

    Returns:
        - Numpy array: The vector of gradient
    """
    grad_loss = np.zeros(len(thetas))
    for i in range(0, len(thetas)):
        
        thetas1, thetas2 = thetas.copy(), thetas.copy()
        thetas1[i] += s
        thetas2[i] -= s

        qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
        qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)

        grad_loss[i] = -r*(
            qtm.base_qtm.measure(qc1, range(qc1.num_qubits)) - 
            qtm.base_qtm.measure(qc2, range(qc2.num_qubits))
        )
    return grad_loss

def grad_psi(
    qc: qiskit.QuantumCircuit, 
    create_circuit_func: FunctionType, 
    thetas: np.ndarray, r: float, s: float, **kwargs):
    """Return the derivatite of the psi base on parameter shift rule

    Args:
        - qc (qiskit.QuantumCircuit): [description]
        - create_circuit_func (FunctionType): [description]
        - thetas (np.ndarray): [description]
        - r (float): in psr
        - s (float): in psr

    Returns:
        - np.ndarray: N x N matrix
    """
    gradient_psi = []
    for i in range(0, len(thetas)):
        thetas_copy = thetas.copy()
        thetas_copy[i] += s
        qc1 = create_circuit_func(qc.copy(), thetas_copy, **kwargs)
        psi_qc = qiskit.quantum_info.Statevector.from_instruction(qc1).data
        psi_qc = np.expand_dims(psi_qc, 1)
        
        gradient_psi.append(r*psi_qc)

    gradient_psi = np.array(gradient_psi)
    return gradient_psi

def loss_basis(measurement_value: float):
    """Return loss value for loss function L = 1 - P_0
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (Float): P_0 value

    Returns:
        - Float: Loss value
    """
    return 1 - measurement_value

def sgd(thetas: np.ndarray, grad_loss):
    """Standard gradient descent

    Args:
        - thetas (np.ndarray): params
        - grad_loss (float): gradient value

    Returns:
        - np.ndarray: New params
    """
    thetas -= qtm.constant.learning_rate * grad_loss
    return thetas

def adam(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, iteration: int, grad_loss: np.ndarray):
    """Adam Optimizer. Below codes are copied from somewhere :)

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    num_thetas = thetas.shape[0]
    beta1, beta2, epsilon = 0.8, 0.999, 10**(-8)
    for i in range(0, num_thetas):
        m[i] = beta1 * m[i] + (1 - beta1) * grad_loss[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad_loss[i]**2
        mhat = m[i] / (1 - beta1**(iteration + 1))
        vhat = v[i] / (1 - beta2**(iteration + 1))
        thetas[i] -= qtm.constant.learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    return thetas

def qng(thetas: np.ndarray, psi: np.ndarray, grad_psi: np.ndarray, grad_loss: np.ndarray):
    """Update parameters based on quantum natural gradient algorithm
    \n thetas^{i + 1} = thetas^{i} - alpha * F^{-1} * nabla L

    Args:
        - thetas (np.ndarray): parameters
        - psi (np.ndarray): current state 
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    F = qtm.quantum_fisher.create_QFIM(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.linalg.pinv(F, hermitian = True)
    else:
        inverse_F = np.linalg.inv(F)

    # inverse_F = np.linalg.pinv(F, hermitian = True)
    
    thetas -= qtm.constant.learning_rate*np.dot(inverse_F, grad_loss)
    return thetas

def qng_adam(thetas: np.ndarray, 
    m: np.ndarray, v: np.ndarray, i: int, 
    psi: np.ndarray, grad_psi: np.ndarray, grad_loss: np.ndarray):
    """After calculating the QFIM, use it in Adam optimizer

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - psi (np.ndarray): current state 
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        np.ndarray: parameters after update
    """
    F = qtm.quantum_fisher.create_QFIM(psi, grad_psi)
    inverse_F = np.linalg.pinv(F)
    grad = np.dot(inverse_F, grad_loss)
    thetas = qtm.base_qtm.adam(thetas, m, v, i, grad)
    return thetas

def fit(qc: qiskit.QuantumCircuit, num_steps: int, thetas, 
    create_circuit_func: FunctionType, 
    grad_func: FunctionType, 
    loss_func: FunctionType,
    optimizer: FunctionType,
    verbose: int = 0,
    **kwargs):
    """Return the new thetas that fit with the circuit from create_circuit_func function

    Args:
        - qc (QuantumCircuit): Fitting circuit
        - num_steps (Int): number of iterations
        - thetas (Numpy arrray): Parameters
        - create_circuit_func (FunctionType): Added circuit function
        - grad_func (FunctionType): Gradient function
        - loss_func (FunctionType): Loss function
        - optimizer (FunctionType): Otimizer function
        - verbose (Int): the seeing level of the fitting process (0: nothing, 1: progress bar, 2: one line per step)
        - **kwargs: additional parameters for different create_circuit_func()
    
    Returns:
        - thetas (Numpy array): the optimized parameters
        - loss_values (Numpy array): the list of loss_value
    """

    loss_values = []
    if verbose == 1:
        bar = qtm.progress_bar.ProgressBar(max_value = num_steps, disable = False)   
    for i in range(0, num_steps):     
        grad_loss = grad_func(qc, create_circuit_func, thetas, 1/2, np.pi/2, **kwargs)  
        optimizer_name = optimizer.__name__
        
        if optimizer_name == 'sgd': 
            thetas = sgd(thetas, grad_loss) 

        elif optimizer_name == 'adam':
            if i == 0:
                m, v = list(np.zeros(thetas.shape[0])), list(np.zeros(thetas.shape[0]))
            thetas = adam(thetas, m, v, i, grad_loss)

        elif optimizer_name == 'qng' or optimizer_name == 'qng_adam':
            grad_psi1 = grad_psi(qc, create_circuit_func, thetas, r = 1/2, s = np.pi, **kwargs)
            qc_copy = create_circuit_func(qc.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(qc_copy).data
            psi = np.expand_dims(psi , 1)

            if optimizer_name == 'qng':
                
                thetas = qng(thetas, psi, grad_psi1, grad_loss)

            if optimizer_name == 'qng_adam':
                if i == 0:
                    m, v = list(np.zeros(thetas.shape[0])), list(np.zeros(thetas.shape[0]))
                thetas = qng_adam(thetas, m, v, i, psi, grad_psi1, grad_loss)

        qc_copy = create_circuit_func(qc.copy(), thetas, **kwargs)
        loss = loss_func(qtm.base_qtm.measure(qc_copy, range(qc_copy.num_qubits)))
        loss_values.append(loss)
        if verbose == 1:
            bar.update(1)
        if verbose == 2 and i % 10 == 0:
            print("Step " + str(i) + ": " + str(loss))
    if verbose == 1:      
        bar.close()
    return thetas, loss_values
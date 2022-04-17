import qiskit
import scipy
import qtm.progress_bar
import qtm.constant
import qtm.qfim
import numpy as np
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer import noise
from types import FunctionType
from typing import Dict
from unittest import result


def extract_state(qc: qiskit.QuantumCircuit):
    """Get infomation about quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): Extracted circuit

    Returns:
       - tuple: state vector and density matrix
    """
    psi = qiskit.quantum_info.Statevector.from_instruction(qc)
    rho_psi = qiskit.quantum_info.DensityMatrix(psi)
    return psi, rho_psi


def generate_depolarizing_noise_model(prob: float):
    prob_1 = prob  # 1-qubit gate
    prob_2 = prob  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = qiskit.providers.aer.noise.depolarizing_error(prob_1, 1)
    error_2 = qiskit.providers.aer.noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = qiskit.providers.aer.noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(
        error_2, ['cx', 'cz', 'crx', 'cry', 'crz'])
    return noise_model


def generate_noise_model(num_qubit, error_prob):
    noise_model = noise.NoiseModel()
    for qi in range(num_qubit):
        read_err = noise.errors.readout_error.ReadoutError(
            [[1 - error_prob, error_prob], [error_prob, 1 - error_prob]])
        noise_model.add_readout_error(read_err, [qi])
    return noise_model


def generate_measurement_filter(num_qubits, noise_model):
    # for running measurement error mitigation
    meas_cals, state_labels = complete_meas_cal(qubit_list=range(
        num_qubits), qr=qiskit.QuantumRegister(num_qubits))
    # Execute the calibration circuits
    job = qiskit.execute(meas_cals, backend=qtm.constant.backend,
                         shots=qtm.constant.num_shots, noise_model=noise_model)
    cal_results = job.result()
    # Make a calibration matrix
    meas_filter = CompleteMeasFitter(cal_results, state_labels).filter
    return meas_filter


def measure(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """Measuring the quantu circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00.. cbit
    """
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    if qtm.constant.noise_prob > 0:
        noise_model = generate_noise_model(n, qtm.constant.noise_prob)
        results = qiskit.execute(qc, backend=qtm.constant.backend,
                                 noise_model=noise_model,
                                 shots=qtm.constant.num_shots).result()
        # Raw counts
        counts = results.get_counts()
        # Mitigating noise based on https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
        meas_filter = generate_measurement_filter(n, noise_model=noise_model)
        # Mitigated counts
        counts = meas_filter.apply(counts.copy())
    else:
        counts = qiskit.execute(
            qc, backend=qtm.constant.backend,
            shots=qtm.constant.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / qtm.constant.num_shots


def x_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.h(qubits[i])
        qc.measure(qubits[i], cbits[i])
    return qc


def y_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.sdg(qubits[i])
        qc.h(qubits[i])
        qc.measure(qubits[i], cbits[i])
    return qc


def z_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], cbits[i])
    return qc


def trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1 / 2 * sum(abs(w))


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
    return np.trace(
        scipy.linalg.sqrtm(
            (scipy.linalg.sqrtm(rho)) @ (rho)) @ (scipy.linalg.sqrtm(sigma)))


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
    return qtm.base.trace_distance(rho,
                                   sigma), qtm.base.trace_fidelity(rho, sigma)


def get_u_hat(thetas, create_circuit_func: FunctionType, num_qubits: int,
              **kwargs):
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


def get_cry_index(create_circuit_func: FunctionType, thetas, num_qubits, **kwargs):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        func (FunctionType): The creating circuit function
        thetas (Numpy array): Parameters
    Returns:
        - Numpy array: The index list has length equal with number of parameters
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    qc = create_circuit_func(qc, thetas, **kwargs)
    layers = qtm.fubini_study.split_into_layers(qc)
    index_list = []
    for layer in layers:
        for gate in layer[1]:
            if gate[0] == 'cry':
                index_list.append(1)
            else:
                index_list.append(0)
            if len(index_list) == len(thetas):
                return index_list
    return index_list


def grad_loss(qc: qiskit.QuantumCircuit, create_circuit_func: FunctionType,
              thetas, **kwargs):
    """Return the gradient of the loss function

    L = 1 - |<psi~|psi>|^2 = 1 - P_0

    => nabla_L = - nabla_P_0 = - r (P_0(+s) - P_0(-s))

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - create_circuit_func (Function): The creating circuit function
        - thetas (Numpy array): Parameters
        - c_0 (float): cost value
        - **kwargs: additional parameters for different create_circuit_func()

    Returns:
        - Numpy array: The vector of gradient
    """
    index_list = get_cry_index(create_circuit_func, thetas,
                               num_qubits=qc.num_qubits, **kwargs)
    grad_loss = np.zeros(len(thetas))
    alpha = np.pi / 2
    beta = 3 * np.pi / 2
    d_plus = (np.sqrt(2) + 1) / (4*np.sqrt(2))
    d_minus = (np.sqrt(2) - 1) / (4*np.sqrt(2))
    for i in range(0, len(thetas)):
        if index_list[i] == 0:
            # In equation (13)
            thetas1, thetas2 = thetas.copy(), thetas.copy()
            thetas1[i] += np.pi/2
            thetas2[i] -= np.pi/2

            qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
            qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)

            grad_loss[i] = -1/2 * (
                qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
                qtm.base.measure(qc2, list(range(qc2.num_qubits))))
        if index_list[i] == 1:
            # In equation (14)
            thetas1, thetas2 = thetas.copy(), thetas.copy()
            thetas3, thetas4 = thetas.copy(), thetas.copy()
            thetas1[i] += alpha
            thetas2[i] -= alpha
            thetas3[i] += beta
            thetas4[i] -= beta
            qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
            qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)
            qc3 = create_circuit_func(qc.copy(), thetas3, **kwargs)
            qc4 = create_circuit_func(qc.copy(), thetas4, **kwargs)
            grad_loss[i] = - (d_plus * (
                qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
                qtm.base.measure(qc2, list(range(qc2.num_qubits)))) - d_minus * (
                qtm.base.measure(qc3, list(range(qc3.num_qubits))) -
                qtm.base.measure(qc4, list(range(qc4.num_qubits)))))
    return grad_loss


def grad_psi(qc: qiskit.QuantumCircuit, create_circuit_func: FunctionType,
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

        gradient_psi.append(r * psi_qc)

    gradient_psi = np.array(gradient_psi)
    return gradient_psi


def loss_basis(measurement_value: Dict[str, int]):
    """Return loss value for loss function L = 1 - P_0
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (Float): P_0 value

    Returns:
        - Float: Loss value
    """
    return 1 - measurement_value


def loss_fubini_study(measurement_value: Dict[str, int]):
    """Return loss value for loss function C = (1 - P_0)^(1/2)
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (Float): P_0 value

    Returns:
        - Float: Loss value
    """
    return np.sqrt(1 - measurement_value)


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


def adam(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, iteration: int,
         grad_loss: np.ndarray):
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
        thetas[i] -= qtm.constant.learning_rate * mhat / (np.sqrt(vhat) +
                                                          epsilon)
    return thetas


def qng_fubini_study(thetas: np.ndarray, G: np.ndarray, grad_loss: np.ndarray):
    """_summary_

    Args:
        - thetas (np.ndarray): parameters
        - G (np.ndarray): Fubini-study matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    thetas = np.real(thetas - qtm.constant.learning_rate *
                     (np.linalg.inv(G) @ grad_loss))
    return thetas


def qng_qfim(thetas: np.ndarray, psi: np.ndarray, grad_psi: np.ndarray,
             grad_loss: np.ndarray):
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
    F = qtm.qfim.create_QFIM(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero

    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.inv(F)
    thetas -= qtm.constant.learning_rate * (inverse_F @ grad_loss)
    return thetas


def qng_adam(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, i: int,
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
    F = qtm.qfim.create_QFIM(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.inv(F)

    grad = inverse_F @ grad_loss
    thetas = qtm.base.adam(thetas, m, v, i, grad)
    return thetas


def fit(qc: qiskit.QuantumCircuit,
        num_steps: int,
        thetas,
        create_circuit_func: FunctionType,
        grad_func: FunctionType,
        loss_func: FunctionType,
        optimizer: FunctionType,
        verbose: int = 0,
        is_return_all_thetas: bool = False,
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
    thetass = []
    loss_values = []
    if verbose == 1:
        bar = qtm.progress_bar.ProgressBar(max_value=num_steps, disable=False)
    for i in range(0, num_steps):
        grad_loss = grad_func(qc, create_circuit_func, thetas, 1 / 2,
                              np.pi / 2, **kwargs)
        optimizer_name = optimizer.__name__

        if optimizer_name == 'sgd':
            thetas = sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            thetas = adam(thetas, m, v, i, grad_loss)

        elif optimizer_name in ['qng_fubini_study', 'qng_qfim', 'qng_adam']:
            grad_psi1 = grad_psi(qc,
                                 create_circuit_func,
                                 thetas,
                                 r=1 / 2,
                                 s=np.pi,
                                 **kwargs)
            qc_copy = create_circuit_func(qc.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(
                qc_copy).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(
                    qc.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_qfim':

                thetas = qng_qfim(thetas, psi, grad_psi1, grad_loss)

            if optimizer_name == 'qng_adam':
                if i == 0:
                    m, v = list(np.zeros(thetas.shape[0])), list(
                        np.zeros(thetas.shape[0]))
                thetas = qng_adam(thetas, m, v, i, psi, grad_psi1, grad_loss)

        qc_copy = create_circuit_func(qc.copy(), thetas, **kwargs)
        loss = loss_func(
            qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
        loss_values.append(loss)
        thetass.append(thetas.copy())
        if verbose == 1:
            bar.update(1)
        if verbose == 2 and i % 10 == 0:
            print("Step " + str(i) + ": " + str(loss))
    if verbose == 1:
        bar.close()

    if is_return_all_thetas:
        return thetass, loss_values
    else:
        return thetas, loss_values

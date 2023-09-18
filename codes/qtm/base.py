import qiskit
import qtm.progress_bar
import qtm.constant
import qtm.qfim
import qtm.noise
import qtm.optimizer
import qtm.fubini_study
import qtm.psr
import numpy as np
import types
import typing
import tensorflow as tf
import qtm.constant
import qtm.early_stopping
global thetas_1


def measure(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """Measuring the quantu circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (np.ndarray): List of measured qubit

    Returns:
        - float: Frequency of 00.. cbit
    """
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    if qc.num_clbits == 0:
        cr = qiskit.ClassicalRegister(qc.num_qubits, 'c')
        qc.add_register(cr)
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    # qc.measure_all() # 
    if qtm.constant.noise_prob > 0:
        noise_model = qtm.noise.generate_noise_model(
            n, qtm.constant.noise_prob)
        results = qiskit.execute(qc, backend=qtm.constant.backend,
                                 noise_model=noise_model,
                                 shots=qtm.constant.num_shots).result()
        # Raw counts
        counts = results.get_counts()
        # Mitigating noise based on https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
        meas_filter = qtm.noise.generate_measurement_filter(
            n, noise_model=noise_model)
        # # Mitigated counts
        counts = meas_filter.apply(counts.copy())
    else:
        counts = qiskit.execute(
            qc, backend=qtm.constant.backend,
            shots=qtm.constant.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / qtm.constant.num_shots


def x_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """As its function name

    Args:
        qc (qiskit.QuantumCircuit): measuremed circuit
        qubits (np.ndarray): list of measuremed qubit
        cbits (list, optional): classical bits. Defaults to [].

    Returns:
        qiskit.QuantumCircuit: added measure gates circuit
    """
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.h(qubits[i])
        qc.measure(qubits[i], cbits[i])
    return qc


def y_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """As its function name

    Args:
        qc (qiskit.QuantumCircuit): measuremed circuit
        qubits (np.ndarray): list of measuremed qubit
        cbits (list, optional): classical bits. Defaults to [].

    Returns:
        qiskit.QuantumCircuit: added measure gates circuit
    """
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.sdg(qubits[i])
        qc.h(qubits[i])
        qc.measure(qubits[i], cbits[i])
    return qc


def z_measurement(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """As its function name

    Args:
        qc (qiskit.QuantumCircuit): measuremed circuit
        qubits (np.ndarray): list of measuremed qubit
        cbits (list, optional): classical bits. Defaults to [].

    Returns:
        qiskit.QuantumCircuit: added measure gates circuit
    """
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], cbits[i])
    return qc


def get_u_hat(thetas: np.ndarray, create_circuit_func: types.FunctionType, num_qubits: int,
              **kwargs):
    """Return inverse of reconstructed gate

    Args:
        - thetas (np.ndarray): Parameters
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


def get_cry_index(qc, thetas):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        - func (types.FunctionType): The creating circuit function
        - thetas (np.ndarray): Parameters
    Returns:
        - np.ndarray: The index list has length equal with number of parameters
    """
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


def grad_loss(qc: qiskit.QuantumCircuit,
              thetas: np.ndarray):
    """Return the gradient of the loss function

    L = 1 - |<psi~|psi>|^2 = 1 - P_0

    => nabla_L = - nabla_P_0 = - r (P_0(+s) - P_0(-s))

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - create_circuit_func (Function): The creating circuit function
        - thetas (np.ndarray): Parameters
        - c_0 (float): cost value
        - **kwargs: additional parameters for different create_circuit_func()

    Returns:
        - np.ndarray: the gradient vector
    """
    index_list = get_cry_index(qc, thetas)
    grad_loss = np.zeros(len(thetas))

    for i in range(0, len(thetas)):
        if index_list[i] == 0:
            # In equation (13)
            grad_loss[i] = qtm.psr.single_2term_psr(qc, thetas, i)
        if index_list[i] == 1:
            # In equation (14)
            grad_loss[i] = qtm.psr.single_4term_psr(qc, thetas, i)
    return grad_loss


def grad_psi(qc: qiskit.QuantumCircuit, thetas: np.ndarray, r: float, s: float):
    """Return the derivatite of the psi base on parameter shift rule

    Args:
        - qc (qiskit.QuantumCircuit): circuit
        - thetas (np.ndarray): parameters
        - r (float): in psr
        - s (float): in psr

    Returns:
        - np.ndarray: N x N matrix
    """
    gradient_psi = []
    for i in range(0, len(thetas)):
        thetas_copy = thetas.copy()
        thetas_copy[i] += s
        qc_copy = qc.bind_parameters(thetas_copy)
        psi_qc = qiskit.quantum_info.Statevector.from_instruction(qc_copy).data
        psi_qc = np.expand_dims(psi_qc, 1)
        gradient_psi.append(r * psi_qc)
    gradient_psi = np.array(gradient_psi)
    return gradient_psi


def fit_state_preparation(u: types.FunctionType,
                          vdagger: qiskit.QuantumCircuit,
                          thetas: np.ndarray,
                          num_steps: int,
                          loss_func: types.FunctionType,
                          optimizer: types.FunctionType,
                          verbose: int = 0,
                          is_return_all_thetas: bool = False,
                          **kwargs):
    """Return the new thetas that fit with the circuit from create_u_func function

    Args:
        - create_u_func (types.FunctionType): added circuit function
        - vdagger (QuantumCircuit): fitting circuit
        - thetas (np.ndarray): parameters
        - num_steps (Int): number of iterations
        - loss_func (types.FunctionType): loss function
        - optimizer (types.FunctionType): otimizer function
        - verbose (Int): the seeing level of the fitting process (0: nothing, 1: progress bar, 2: one line per step)
        - **kwargs: additional parameters for create_circuit_func()

    Returns:
        - thetas (np.ndarray): the optimized parameters
        - loss_values (np.ndarray): the list of loss_value
    """
    if verbose == 1:
        bar = qtm.progress_bar.ProgressBar(max_value=num_steps, disable=False)
    thetass = []
    loss_values = []
    n = vdagger.num_qubits
    uvaddager = u.compose(vdagger)
    for i in range(0, num_steps):
        grad_loss = qtm.base.grad_loss(uvaddager, thetas)
        optimizer_name = optimizer.__name__

        if optimizer_name == 'sgd':
            thetas = qtm.optimizer.sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v1 = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            thetas = qtm.optimizer.adam(thetas, m, v1, i, grad_loss)

        elif 'qng' in optimizer_name:
            grad_psi1 = qtm.base.grad_psi(uvaddager, thetas,
                                          r=1 / 2,
                                          s=np.pi)
            qc_binded = uvaddager.bind_parameters(thetas)
            psi = qiskit.quantum_info.Statevector.from_instruction(qc_binded).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(uvaddager)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_fubini_hessian':
                G = qtm.fubini_study.qng_hessian(uvaddager)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_fubini_study_scheduler':
                G = qtm.fubini_study.qng(uvaddager)
                thetas = qtm.optimizer.qng_fubini_study_scheduler(
                    thetas, G, grad_loss, i)
            if optimizer_name == 'qng_qfim':

                thetas = qtm.optimizer.qng_qfim(
                    thetas, psi, grad_psi1, grad_loss)

            if optimizer_name == 'qng_adam':
                if i == 0:
                    m, v1 = list(np.zeros(thetas.shape[0])), list(
                        np.zeros(thetas.shape[0]))
                thetas = qtm.optimizer.qng_adam(
                    thetas, m, v1, i, psi, grad_psi1, grad_loss)
        else:
            thetas = optimizer(thetas, grad_loss)
        
        qc_binded = uvaddager.bind_parameters(thetas)
        loss = loss_func(
            qtm.base.measure(qc_binded, list(range(n))))
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


def fit(u: typing.Union[qiskit.QuantumCircuit, types.FunctionType], v: typing.Union[qiskit.QuantumCircuit, types.FunctionType],
        thetas: np.ndarray,
        num_steps: int,
        loss_func: types.FunctionType,
        optimizer: types.FunctionType,
        verbose: int = 0,
        is_return_all_thetas: bool = False,
        **kwargs):
    return fit_state_preparation(u=u,
                                 vdagger=v,
                                 thetas=thetas,
                                 num_steps=num_steps,
                                 loss_func=loss_func,
                                 optimizer=optimizer,
                                 verbose=verbose,
                                 is_return_all_thetas=is_return_all_thetas,
                                 **kwargs)
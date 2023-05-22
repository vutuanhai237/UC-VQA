import qiskit
import qtm.progress_bar
import qtm.constant
import qtm.qfim
import qtm.noise
import qtm.optimizer
import qtm.fubini_study
import numpy as np
import types
import typing
from DQASsearch import DQAS_search
from utils import set_op_pool
from vag import GHZ_vag
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
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    if qtm.constant.noise_prob > 0:
        noise_model = qtm.noise.generate_noise_model(
            n, qtm.constant.noise_prob)
        results = qiskit.execute(qc, backend=qtm.constant.backend,
                                 noise_model=noise_model,
                                 shots=qtm.constant.num_shots).result()
        # Raw counts
        counts = results.get_counts()
        # Mitigating noise based on https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
        # meas_filter = qtm.noise.generate_measurement_filter(
        #     n, noise_model=noise_model)
        # # Mitigated counts
        # counts = meas_filter.apply(counts.copy())
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


def get_cry_index(create_circuit_func: types.FunctionType, thetas: np.ndarray, num_qubits, **kwargs):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        - func (types.FunctionType): The creating circuit function
        - thetas (np.ndarray): Parameters
    Returns:
        - np.ndarray: The index list has length equal with number of parameters
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


def grad_loss(qc: qiskit.QuantumCircuit, create_circuit_func: types.FunctionType,
              thetas: np.ndarray, **kwargs):
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
    index_list = get_cry_index(create_circuit_func, thetas,
                               num_qubits=qc.num_qubits, **kwargs)
    grad_loss = np.zeros(len(thetas))

    for i in range(0, len(thetas)):
        if index_list[i] == 0:
            # In equation (13)
            thetas1, thetas2 = thetas.copy(), thetas.copy()
            thetas1[i] += qtm.constant.two_term_psr['s']
            thetas2[i] -= qtm.constant.two_term_psr['s']

            qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
            qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)

            grad_loss[i] = -qtm.constant.two_term_psr['r'] * (
                qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
                qtm.base.measure(qc2, list(range(qc2.num_qubits))))
        if index_list[i] == 1:
            # In equation (14)
            thetas1, thetas2 = thetas.copy(), thetas.copy()
            thetas3, thetas4 = thetas.copy(), thetas.copy()
            thetas1[i] += qtm.constant.four_term_psr['alpha']
            thetas2[i] -= qtm.constant.four_term_psr['alpha']
            thetas3[i] += qtm.constant.four_term_psr['beta']
            thetas4[i] -= qtm.constant.four_term_psr['beta']
            qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
            qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)
            qc3 = create_circuit_func(qc.copy(), thetas3, **kwargs)
            qc4 = create_circuit_func(qc.copy(), thetas4, **kwargs)
            grad_loss[i] = - (qtm.constant.four_term_psr['d_plus'] * (
                qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
                qtm.base.measure(qc2, list(range(qc2.num_qubits)))) - qtm.constant.four_term_psr['d_minus'] * (
                qtm.base.measure(qc3, list(range(qc3.num_qubits))) -
                qtm.base.measure(qc4, list(range(qc4.num_qubits)))))
    return grad_loss


def grad_psi(qc: qiskit.QuantumCircuit, create_circuit_func: types.FunctionType,
             thetas: np.ndarray, r: float, s: float, **kwargs):
    """Return the derivatite of the psi base on parameter shift rule

    Args:
        - qc (qiskit.QuantumCircuit): circuit
        - create_circuit_func (types.FunctionType)
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
        qc_copy = create_circuit_func(qc.copy(), thetas_copy, **kwargs)
        psi_qc = qiskit.quantum_info.Statevector.from_instruction(qc_copy).data
        psi_qc = np.expand_dims(psi_qc, 1)
        gradient_psi.append(r * psi_qc)
    gradient_psi = np.array(gradient_psi)
    return gradient_psi


def fit_state_tomography(u: qiskit.QuantumCircuit,
                         create_vdagger_func: types.FunctionType,
                         thetas: np.ndarray,
                         num_steps: int,
                         loss_func: types.FunctionType,
                         optimizer: types.FunctionType,
                         verbose: int = 0,
                         is_return_all_thetas: bool = False,
                         **kwargs):
    """Return the new thetas that fit with the circuit from create_vdagger_func function

    Args:
        - u (QuantumCircuit): fitting circuit
        - create_vdagger_func (types.FunctionType): added circuit function
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
    thetass = []
    loss_values = []
    if verbose == 1:
        bar = qtm.progress_bar.ProgressBar(max_value=num_steps, disable=False)
    for i in range(0, num_steps):
        grad_loss = qtm.base.grad_loss(
            u, create_vdagger_func, thetas, **kwargs)
        optimizer_name = optimizer.__name__

        if optimizer_name == 'sgd':
            thetas = qtm.optimizer.sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            thetas = qtm.optimizer.adam(thetas, m, v, i, grad_loss)

        elif 'qng' in optimizer_name:
            grad_psi1 = grad_psi(u,
                                 create_vdagger_func,
                                 thetas,
                                 r=qtm.constant.two_term_psr['s'],
                                 s=np.pi,
                                 **kwargs)
            u_copy = create_vdagger_func(u.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(u_copy).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(
                    u.copy(), thetas, create_vdagger_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_fubini_study_scheduler':
                G = qtm.fubini_study.qng(
                    u.copy(), thetas, create_vdagger_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study_scheduler(thetas, G, grad_loss, i)
            if optimizer_name == 'qng_qfim':
                thetas = qtm.optimizer.qng_qfim(
                    thetas, psi, grad_psi1, grad_loss)
            if optimizer_name == 'qng_adam':
                if i == 0:
                    m, v = list(np.zeros(thetas.shape[0])), list(
                        np.zeros(thetas.shape[0]))
                thetas = qtm.optimizer.qng_adam(
                    thetas, m, v, i, psi, grad_psi1, grad_loss)
        else:
            thetas = optimizer(thetas, grad_loss)
        u_copy = create_vdagger_func(u.copy(), thetas, **kwargs)
        loss = loss_func(
            qtm.base.measure(u_copy, list(range(u_copy.num_qubits))))
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


def fit_state_preparation(create_u_func: types.FunctionType,
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
    def create_circuit_func(vdagger: qiskit.QuantumCircuit, thetas: np.ndarray, **kwargs):
        return create_u_func(qiskit.QuantumCircuit(n, n), thetas, **kwargs).combine(vdagger)
    for i in range(0, num_steps):
        grad_loss = qtm.base.grad_loss(
            vdagger, create_circuit_func, thetas, **kwargs)
        optimizer_name = optimizer.__name__

        if optimizer_name == 'sgd':
            thetas = qtm.optimizer.sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v1 = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            thetas = qtm.optimizer.adam(thetas, m, v1, i, grad_loss)

        elif 'qng' in optimizer_name:
            grad_psi1 = grad_psi(vdagger,
                                 create_circuit_func,
                                 thetas,
                                 r=1 / 2,
                                 s=np.pi,
                                 **kwargs)
            v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(
                v_copy).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(
                    vdagger.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_fubini_study_scheduler':
                G = qtm.fubini_study.qng(
                    vdagger.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study_scheduler(thetas, G, grad_loss, i)
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
        v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
        loss = loss_func(
            qtm.base.measure(v_copy, list(range(n))))
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


def fit_state_preparation_evo(create_u_func: types.FunctionType,
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
    traces = []
    fidelities = []
    thetass = []
    loss_values = []
    is_evo = False
    create_u1_func = None
    n = vdagger.num_qubits
    def create_circuit_func(vdagger: qiskit.QuantumCircuit, _thetas: np.ndarray, **kwargs):
        # nonlocal is_evo
        if create_u1_func is not None:
            return create_u_func(
                qiskit.QuantumCircuit(n, n), _thetas, **kwargs).combine(create_u1_func(qiskit.QuantumCircuit(n, n), thetas_1)).combine(vdagger)
        return create_u_func(qiskit.QuantumCircuit(n, n), _thetas, **kwargs).combine(vdagger)

    for i in range(0, num_steps):
        grad_loss = qtm.base.grad_loss(
            vdagger, create_circuit_func, thetas, **kwargs)
        optimizer_name = optimizer.__name__
        
        if optimizer_name == 'sgd':
            thetas = qtm.optimizer.sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v1 = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            thetas = qtm.optimizer.adam(thetas, m, v1, i, grad_loss)

        elif 'qng' in optimizer_name:
            grad_psi1 = grad_psi(vdagger,
                                 create_circuit_func,
                                 thetas,
                                 r=1 / 2,
                                 s=np.pi,
                                 **kwargs)
            v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(
                v_copy).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(
                    vdagger.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
            if optimizer_name == 'qng_fubini_study_scheduler':
                G = qtm.fubini_study.qng(
                    vdagger.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study_scheduler(thetas, G, grad_loss, i)
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
        v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
        loss = loss_func(
            qtm.base.measure(v_copy, list(range(n))))
        loss_values.append(loss)

        # if evo_condition(loss_values):
        #     set_op_pool(qtm.constant.ghz_pool)
        #     c = len(qtm.constant.ghz_pool)
        #     p = 5
        #     cand_weight, qcircuit = DQAS_search(
        #         GHZ_vag,
        #         nq=3,
        #         p=p,
        #         batch=10,
        #         epochs=3,
        #         verbose=False,
        #         nnp_initial_value=np.zeros([p, c]),
        #         structure_opt=tf.keras.optimizers.Adam(learning_rate=0.15),
        #     )

        #     is_evo = False
        #     thetas_1 = cand_weight
        #     create_u1_func = qcircuit

        trace, fidelity = qtm.utilities.calculate_state_preparation_metrics_tiny(
            create_u_func, vdagger, thetas, **kwargs)
        thetass.append(thetas.copy())
        traces.append(trace)
        fidelities.append(fidelity)
        if verbose == 1:
            bar.update(1)
        if verbose == 2 and i % 10 == 0:
            print("Step " + str(i) + ": " + str(loss))

    if verbose == 1:
        bar.close()

    if is_return_all_thetas:
        return thetass, loss_values, traces, fidelities
    else:
        return thetas, loss_values, traces, fidelities


def fit_state_preparation_evo2(create_u_func: types.FunctionType,
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
    traces = []
    fidelities = []
    thetass = []
    loss_values = []
    early_stopper = qtm.early_stopping.EarlyStopping(patience = 10, delta = qtm.constant.delta)
    additional_circuit = None
    origin_len = thetas.shape[0]
    num_cry_layer = 0
    ces = []
    n = vdagger.num_qubits
    def create_circuit_func(vdagger: qiskit.QuantumCircuit, _thetas: np.ndarray, **kwargs):
        # nonlocal is_evo
        if additional_circuit is not None:
            return create_u_func(
                qiskit.QuantumCircuit(n, n), _thetas[:-num_cry_layer*n], **kwargs).combine(additional_circuit(n, num_cry_layer, _thetas[-num_cry_layer*n:])).combine(vdagger)
        return create_u_func(qiskit.QuantumCircuit(n, n), _thetas, **kwargs).combine(vdagger)

    for i in range(0, num_steps):
        j = 5
        # 0.15, 0.428
        if early_stopper.mode == "active":
            num_cry_layer += 1
            additional_circuit = qtm.ansatz.ry_layer
            thetas = np.append(thetas, [0]*(num_cry_layer*n - (thetas.shape[0]-origin_len)))
            v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
            # print(v_copy.draw())
            early_stopper.mode = "inactive"
                
        grad_loss = qtm.base.grad_loss(
            vdagger, create_circuit_func, thetas, **kwargs)
        optimizer_name = optimizer.__name__

        if optimizer_name == 'sgd':
            thetas = qtm.optimizer.sgd(thetas, grad_loss)

        elif optimizer_name == 'adam':
            if i == 0:
                m, v1 = list(np.zeros(thetas.shape[0])), list(
                    np.zeros(thetas.shape[0]))
            if len(m) < thetas.shape[0] or len(v1) < thetas.shape[0]:
                m += [0]*(thetas.shape[0]-len(m))
                v1 += [0]*(thetas.shape[0]-len(v1))
            thetas = qtm.optimizer.adam(thetas, m, v1, i, grad_loss)

        elif 'qng' in optimizer_name:
            grad_psi1 = grad_psi(vdagger,
                                 create_circuit_func,
                                 thetas,
                                 r=1 / 2,
                                 s=np.pi,
                                 **kwargs)
            v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
            psi = qiskit.quantum_info.Statevector.from_instruction(
                v_copy).data
            psi = np.expand_dims(psi, 1)
            if optimizer_name == 'qng_fubini_study':
                G = qtm.fubini_study.qng(
                    vdagger.copy(), thetas, create_circuit_func, **kwargs)
                thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
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
        v_copy = create_circuit_func(vdagger.copy(), thetas, **kwargs)
        loss = loss_func(
            qtm.base.measure(v_copy, list(range(n))))
        loss_values.append(loss)
        if additional_circuit == None:
            trace, fidelity = qtm.utilities.calculate_state_preparation_metrics_tiny(
                create_u_func, vdagger, thetas, **kwargs)
            
        else:
            trace, fidelity = qtm.utilities.calculate_state_preparation_metrics_tiny2(
                create_u_func, additional_circuit(n, num_cry_layer, thetas[-num_cry_layer*n:]), vdagger, thetas[:-num_cry_layer*n], **kwargs)
        
        thetass.append(thetas.copy())
        traces.append(trace)
        fidelities.append(fidelity)
        u = create_circuit_func(qiskit.QuantumCircuit(n, n), thetass[-1], **kwargs)
        ces.append(qtm.utilities.concentratable_entanglement(u))
        if i > 10:
            early_stopper.track(loss_values[-2], loss_values[-1])
        if verbose == 1:
            bar.update(1)
        if verbose == 2 and i % 10 == 0:
            print("Step " + str(i) + ": " + str(loss))

    if verbose == 1:
        bar.close()

    if is_return_all_thetas:
        return thetass, loss_values, traces, fidelities, ces
    else:
        return thetas, loss_values, traces, fidelities, ces
    
def fit_evo(u: typing.Union[qiskit.QuantumCircuit, types.FunctionType], v: typing.Union[qiskit.QuantumCircuit, types.FunctionType],
            thetas: np.ndarray,
            num_steps: int,
            loss_func: types.FunctionType,
            optimizer: types.FunctionType,
            verbose: int = 0,
            is_return_all_thetas: bool = False,
            **kwargs):
    if callable(u):
        return fit_state_preparation_evo2(create_u_func=u,
                                         vdagger=v,
                                         thetas=thetas,
                                         num_steps=num_steps,
                                         loss_func=loss_func,
                                         optimizer=optimizer,
                                         verbose=verbose,
                                         is_return_all_thetas=is_return_all_thetas,
                                         **kwargs)
    return


def fit(u: typing.Union[qiskit.QuantumCircuit, types.FunctionType], v: typing.Union[qiskit.QuantumCircuit, types.FunctionType],
        thetas: np.ndarray,
        num_steps: int,
        loss_func: types.FunctionType,
        optimizer: types.FunctionType,
        verbose: int = 0,
        is_return_all_thetas: bool = False,
        **kwargs):
    if callable(u):
        return fit_state_preparation(create_u_func=u,
                                     vdagger=v,
                                     thetas=thetas,
                                     num_steps=num_steps,
                                     loss_func=loss_func,
                                     optimizer=optimizer,
                                     verbose=verbose,
                                     is_return_all_thetas=is_return_all_thetas,
                                     **kwargs)
    else:
        return fit_state_tomography(u=u,
                                    create_vdagger_func=v,
                                    thetas=thetas,
                                    num_steps=num_steps,
                                    loss_func=loss_func,
                                    optimizer=optimizer,
                                    verbose=verbose,
                                    is_return_all_thetas=is_return_all_thetas,
                                    **kwargs)

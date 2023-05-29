import qtm.constant
import qiskit
import types
import numpy as np


def single_2term_psr(qc: qiskit.QuantumCircuit, create_circuit_func: types.FunctionType,
                     thetas: np.ndarray, i, **kwargs):
    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas1[i] += qtm.constant.two_term_psr['s']
    thetas2[i] -= qtm.constant.two_term_psr['s']

    qc1 = create_circuit_func(qc.copy(), thetas1, **kwargs)
    qc2 = create_circuit_func(qc.copy(), thetas2, **kwargs)
    return -qtm.constant.two_term_psr['r'] * (
        qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
        qtm.base.measure(qc2, list(range(qc2.num_qubits))))


def single_4term_psr(qc: qiskit.QuantumCircuit, create_circuit_func: types.FunctionType,
                     thetas: np.ndarray, i, **kwargs):
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
    return - (qtm.constant.four_term_psr['d_plus'] * (
        qtm.base.measure(qc1, list(range(qc1.num_qubits))) -
        qtm.base.measure(qc2, list(range(qc2.num_qubits)))) - qtm.constant.four_term_psr['d_minus'] * (
        qtm.base.measure(qc3, list(range(qc3.num_qubits))) -
        qtm.base.measure(qc4, list(range(qc4.num_qubits)))))

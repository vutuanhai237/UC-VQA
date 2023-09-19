import qiskit
from qiskit.providers.aer import noise
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import qtm.constant
def generate_depolarizing_noise_model(prob: float):
    """As its function name

    Args:
        - prob (float): from 0 to 1

    Returns:
        - qiskit.providers.aer.noise.NoiseModel: new noise model
    """
    prob_1 = prob  # 1-qubit gate
    prob_2 = prob  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(
        error_2, ['cx', 'cz', 'crx', 'cry', 'crz'])
    return noise_model


def generate_noise_model(num_qubit: int, error_prob: float):
    """Create readout noise model

    Args:
        - num_qubit (int): number of qubit
        - error_prob (float):from 0 to 1

    Returns:
        - qiskit.providers.aer.noise.NoiseModel: new noise model
    """
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
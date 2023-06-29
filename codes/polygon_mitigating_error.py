import qiskit, sys
import numpy as np
import qtm.qcompilation, qtm.ansatz, qtm.constant
import matplotlib.pyplot as plt

print(qtm.constant.noise_prob)

time = 3

for i in range(7,8):
    num_qubits = 5
    num_layers = 2
    thetas = np.ones(num_layers*num_qubits*2)

    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.ansatz.create_polygongraph_ansatz,
        vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
        optimizer = 'qng_fubini_study',
        loss_func = 'loss_fubini_study',
        thetas = thetas,
        num_layers = num_layers
    )
    compiler.fit(num_steps=4, verbose = 1)
    compiler.save(path = f'./noise_qng_polygon/{i}', text=f'{qtm.constant.noise_prob}_mitigating', save_all = True)
    print(f"Done {i}!")
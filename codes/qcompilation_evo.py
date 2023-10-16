import qtm

num_qubits = 3
num_layers = 1
compiler = qtm.qcompilation.QuantumCompilation(
    u=qtm.ansatz.create_g_state(num_qubits),
    vdagger=qtm.ansatz.Wchain_ZXZlayer_ansatz(
        num_qubits, num_layers),
    optimizer='adam',
    loss_func='loss_fubini_study'
)

compiler.fit(num_steps=100, verbose=1)



import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.qcompilation


def f(num_qubits, num_layers):
    optimizer = 'adam'
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.ansatz.g2gn(num_qubits, num_layers),
        vdagger = qtm.state.create_ame_state(num_qubits).inverse(),
        optimizer = optimizer,
        loss_func = 'loss_fubini_study'
    )
    compiler.fit(num_steps = 100, verbose = 1)
    qspobj = qtm.qsp.QuantumStatePreparation(
        compiler.u, 
        compiler.vdagger,
        compiler.thetas,
        ansatz = qtm.ansatz.g2gn
    )
    qspobj.save(state = 'ame', file_name='../experiments/qsp/')
    return 
if __name__ == "__main__":
    # creating thread
    list_num_qubits = [3]
    list_num_layers = [1, 2]
   
    processes = [[0 for _ in range(len(list_num_layers))] for _ in range(len(list_num_qubits))]
    for i in range(len(list_num_qubits)):
        for j in range(len(list_num_layers)):
            processes[i][j] = (multiprocessing.Process(target = f, args=(list_num_qubits[i], list_num_layers[j])))

    for i in range(len(list_num_qubits)):
        for j in range(len(list_num_layers)):
            processes[i][j].start()

    for i in range(len(list_num_qubits)):
        for j in range(len(list_num_layers)):
            processes[i][j].join()

    print("Done!")
import sys
import multiprocessing
sys.path.insert(1, '../')
import qtm.qcompilation


def f(num_qubits, num_layers):
    print(num_qubits*num_layers)
    return 
if __name__ == "__main__":
    # creating thread
    list_num_qubits = [3]
    list_num_layers = [1,2,3,4,5]
   
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
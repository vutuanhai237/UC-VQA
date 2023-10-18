import os
import sys
sys.path.insert(0,  '../')
import qtm.qcompilation
import matplotlib.pyplot as plt


def find_satisfying_ansatz(num_qubit, max_depth, num_param ,state_name, error_rate):
    best_fidelity = 0
    best_state = qtm.qsp.QuantumStatePreparation(f"../experiments/qsp/{state_name}_g2_{num_qubit}_1.qspobj")

    for i in ['g2gn','g2','g2gnw']:
        for j in [1,2,3,4,5,6,7,8,9,10]:
            qspobj2 = qtm.qsp.QuantumStatePreparation(f"../experiments/qsp/{state_name}_{i}_{num_qubit}_{j}.qspobj")

            # Compare fidelity
            if qspobj2.fidelity > best_fidelity:
                best_state = qspobj2
                best_fidelity = qspobj2.fidelity
            
            if qspobj2.fidelity == best_fidelity:

                # When fidelity is the same, compare depth
                if qspobj2.u.depth() < best_state.u.depth():
                    best_state = qspobj2

                # When depth is the same, compare num_params
                elif qspobj2.u.depth() == qspobj2.u.depth():
                    if qspobj2.num_params < best_state.num_params:
                        best_state = qspobj2
            
            if best_fidelity > 1 - error_rate:
                return best_state.ansatz
            
def get_thetas(num_qubits, num_layers, state_name, ansatz):
    thetas = qtm.qsp.QuantumStatePreparation(f"../experiments/qsp/{state_name}_{ansatz}_{num_qubits}_{num_layers}.qspobj").thetas

    return thetas

# Test satisfy ansatz
def find_satisfying_ansatz_test(num_qubit, max_depth, num_param ,state_name, error_rate):
    best_fidelity = 0
    best_state = qtm.qsp.QuantumStatePreparation(f"../experiments/qsp/{state_name}_g2_{num_qubit}_1.qspobj")

    for i in ['g2gn','g2','g2gnw']:
        for j in [1,2,3,4,5,6,7,8,9,10]:

            qspobj2 = qtm.qsp.QuantumStatePreparation(f"../experiments/qsp/{state_name}_{i}_{num_qubit}_{j}.qspobj")

            # Compare fidelity
            if qspobj2.fidelity > best_fidelity:

                # Consider error rate
                if qspobj2.fidelity > 1 - error_rate:
                    best_state = qspobj2
                    best_fidelity = qspobj2.fidelity

            if qspobj2.fidelity == best_fidelity:

                # When fidelity is the same, compare depth
                if qspobj2.u.depth() < best_state.u.depth():
                    best_state = qspobj2

                # When depth is the same, compare num_params
                elif qspobj2.u.depth() == qspobj2.u.depth():
                    if qspobj2.num_params < best_state.num_params:
                        best_state = qspobj2
            
            print(best_fidelity,best_state.u.depth(),best_state.num_params)             
            
            if best_fidelity > 1 - error_rate:
                return 




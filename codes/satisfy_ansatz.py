import os
import sys
sys.path.insert(0,  '../')
import qtm.qcompilation

def find_satisfying_ansatz(num_qubit, max_depth, num_param ,state_name):
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

    return best_state.ansatz

print(find_satisfying_ansatz(3,2,2,'AME'))


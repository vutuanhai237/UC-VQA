# import qiskit
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# sys.path.insert(1, '../../')
# import qtm.base, qtm.constant, qtm.ansatz, qtm.gradient, qtm.progress_bar
# import types
# import pickle 

# num_qubits = 3
# num_layers = 2
# # v = qtm.state.create_AME_state(3)
# # prob = qtm.utilities.concentratable_entanglement(v,exact=True)
# # dsd

# # #######################################################
# # thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers*(3*num_qubits)+num_layers*qtm.ansatz.calculate_n_walltoall(num_qubits))])

# # qng_compiler = qtm.qcompilation.QuantumCompilation(
# #     u = qtm.ansatz.create_Walltoall_layered_ansatz,
# #     vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #     optimizer = 'qng_fubini_study',
# #     loss_func = 'loss_fubini_study',
# #     thetas = thetas,
# #     num_layers = num_layers
# # )
# # qng_compiler.fit(num_steps = 100, verbose = 1)

# # qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/Walternating_AME',save_all=True)

# # #######################################################
# # thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range( num_layers * (num_qubits * 5 - 4))])

# # qng_compiler = qtm.qcompilation.QuantumCompilation(
# #     u = qtm.ansatz.create_alternating_layered_ansatz,
# #     vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #     optimizer = 'qng_fubini_study',
# #     loss_func = 'loss_fubini_study',
# #     thetas = thetas,
# #     num_layers = num_layers
# # )
# # qng_compiler.fit(num_steps = 100, verbose = 1)

# # qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/alternating_AME',save_all=True)

# # #######################################################
# # thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range( num_layers * (num_qubits * 4))])

# # qng_compiler = qtm.qcompilation.QuantumCompilation(
# #     u = qtm.ansatz.create_Wchain_layered_ansatz,
# #     vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #     optimizer = 'qng_fubini_study',
# #     loss_func = 'loss_fubini_study',
# #     thetas = thetas,
# #     num_layers = num_layers
# # )
# # qng_compiler.fit(num_steps = 100, verbose = 1)

# # qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/Wchain_AME',save_all=True)

# # ###################################
# # thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * num_qubits * 5)])

# # qng_compiler = qtm.qcompilation.QuantumCompilation(
# #     u = qtm.ansatz.create_linear_ansatz,
# #     vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #     optimizer = 'qng_fubini_study',
# #     loss_func = 'loss_fubini_study',
# #     thetas = thetas,
# #     num_layers = num_layers
# # )
# # qng_compiler.fit(num_steps = 100, verbose = 1)

# # qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/linear',save_all=True)

# ###################################
# # thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (num_qubits * 3))])

# # qng_compiler = qtm.qcompilation.QuantumCompilation(
# #     u = qtm.ansatz.create_WchainCNOT_layered_ansatz,
# #     vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #     optimizer = 'qng_fubini_study',
# #     loss_func = 'loss_fubini_study',
# #     thetas = thetas,
# #     num_layers = num_layers
# # )
# # qng_compiler.fit(num_steps = 100, verbose = 1)

# # qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/WchainCNOT',save_all=True)


# # for i in range(6,10):
# #     ##########################################3
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     qng_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_compiler.fit(num_steps = 100, verbose = 1)
# #     qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/poly_layer_2',save_all=True,run_trial=i)

# # for i in range(0,10):
# #     #########################################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     adam_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #         optimizer = 'adam',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     adam_compiler.fit(num_steps = 100, verbose = 1)


# #     adam_compiler.save(text='adam_poly',path='/home/fptu/tung/UC-VQA/experiments/poly_layer_2',save_all=True,run_trial=i)

# # for i in range(0,10):
# #     #####################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     sgd_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #         optimizer = 'sgd',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     sgd_compiler.fit(num_steps = 100, verbose = 1)


# #     sgd_compiler.save(text='sgd_poly',path='/home/fptu/tung/UC-VQA/experiments/poly_layer_2',save_all=True,run_trial=i)

# # for i in range(0,10):
# #     ##########################################3
# #     thetas =  np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     qng_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_star_compiler.fit(num_steps = 100, verbose = 1)


# #     qng_star_compiler.save(text='qng_star',path='/home/fptu/tung/UC-VQA/experiments/star_layer_2',save_all=True,run_trial=i)

# for i in range(9,10):
#     #####################################
#     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

#     adam_star_compiler = qtm.qcompilation.QuantumCompilation(
#         u = qtm.ansatz.create_stargraph_ansatz,
#         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
#         optimizer = 'adam',
#         loss_func = 'loss_fubini_study',
#         thetas = thetas,
#         num_layers = num_layers
#     )
#     adam_star_compiler.fit(num_steps = 100, verbose = 1)


#     adam_star_compiler.save(text='adam_star',path='/home/fptu/tung/UC-VQA/experiments/star_layer_2',save_all=True,run_trial=i)

# for i in range(0,10):
#     ########################
#     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

#     sgd_star_compiler = qtm.qcompilation.QuantumCompilation(
#         u = qtm.ansatz.create_stargraph_ansatz,
#         vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
#         optimizer = 'sgd',
#         loss_func = 'loss_fubini_study',
#         thetas = thetas,
#         num_layers = num_layers
#     )
#     sgd_star_compiler.fit(num_steps = 100, verbose = 1)


#     sgd_star_compiler.save(text='sgd_star',path='/home/fptu/tung/UC-VQA/experiments/star_layer_2',save_all=True,run_trial=i)

# # for i in range(10):
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     qng_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_compiler.fit(num_steps = 100, verbose = 1)
# #     qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/w_poly_2_layer',save_all=True,run_trial=i)


# # for i in range(10):
# #     #########################################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     adam_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'adam',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     adam_compiler.fit(num_steps = 100, verbose = 1)


# #     adam_compiler.save(text='adam_poly',path='/home/fptu/tung/UC-VQA/experiments/w_poly_2_layer',save_all=True,run_trial=i)

# # for i in range(10):  
# #     #####################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     sgd_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'sgd',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     sgd_compiler.fit(num_steps = 100, verbose = 1)


# #     sgd_compiler.save(text='sgd_poly',path='/home/fptu/tung/UC-VQA/experiments/w_poly_2_layer',save_all=True,run_trial=i)

# # for i in range(10):
# #     ##########################################3
# #     thetas =  np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     qng_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_star_compiler.fit(num_steps = 100, verbose = 1)


# #     qng_star_compiler.save(text='qng_star',path='/home/fptu/tung/UC-VQA/experiments/w_star_2_layer',save_all=True,run_trial=i)

# # for i in range(10):
# #     #####################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     adam_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'adam',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     adam_star_compiler.fit(num_steps = 100, verbose = 1)


# #     adam_star_compiler.save(text='adam_star',path='/home/fptu/tung/UC-VQA/experiments/w_star_2_layer',save_all=True,run_trial=i)

# # for i in range(5,10): 
# #     ########################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     sgd_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_w_state(num_qubits).inverse(),
# #         optimizer = 'sgd',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     sgd_star_compiler.fit(num_steps = 100, verbose = 1)


#     # sgd_star_compiler.save(text='sgd_star',path='/home/fptu/tung/UC-VQA/experiments/w_star_2_layer',save_all=True,run_trial=i)




# # for i in range(10): 

# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     qng_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_compiler.fit(num_steps = 100, verbose = 1)                                       
# #     qng_compiler.save(text='qng_poly',path='/home/fptu/tung/UC-VQA/experiments/ghz_poly_2_layers',save_all=True,run_trial=i)

# # for i in range(10): 

# #     #########################################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     adam_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'adam',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     adam_compiler.fit(num_steps = 100, verbose = 1)


# #     adam_compiler.save(text='adam_poly',path='/home/fptu/tung/UC-VQA/experiments/ghz_poly_2_layers',save_all=True,run_trial=i)

# # for i in range(10): 

# #     #####################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(2*num_qubits*num_layers)])

# #     sgd_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_polygongraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'sgd',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     sgd_compiler.fit(num_steps = 100, verbose = 1)


# #     sgd_compiler.save(text='sgd_poly',path='/home/fptu/tung/UC-VQA/experiments/ghz_poly_2_layers',save_all=True,run_trial=i)

# # for i in range(10): 

# #     ##########################################3
# #     thetas =  np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     qng_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'qng_fubini_study',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     qng_star_compiler.fit(num_steps = 100, verbose = 1)


# #     qng_star_compiler.save(text='qng_star',path='/home/fptu/tung/UC-VQA/experiments/ghz_star_2_layers',save_all=True,run_trial=i)

# # for i in range(10): 

# #     #####################################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     adam_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'adam',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     adam_star_compiler.fit(num_steps = 100, verbose = 1)


# #     adam_star_compiler.save(text='adam_star',path='/home/fptu/tung/UC-VQA/experiments/ghz_star_2_layers',save_all=True,run_trial=i)

# # for i in range(9,10): 

# #     ########################
# #     thetas = np.array([np.random.uniform(0,2*np.pi) for _ in range(num_layers * (2 * num_qubits - 2))])

# #     sgd_star_compiler = qtm.qcompilation.QuantumCompilation(
# #         u = qtm.ansatz.create_stargraph_ansatz,
# #         vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
# #         optimizer = 'sgd',
# #         loss_func = 'loss_fubini_study',
# #         thetas = thetas,
# #         num_layers = num_layers
# #     )
# #     sgd_star_compiler.fit(num_steps = 100, verbose = 1)


# #     sgd_star_compiler.save(text='sgd_star',path='/home/fptu/tung/UC-VQA/experiments/ghz_star_2_layers',save_all=True,run_trial=i)

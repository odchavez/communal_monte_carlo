#parameter_history_obj.compile_bo_list_history(name_stem.code)
    
#parmas_shape = parameter_history_obj.bo_list_history.shape
#print("parmas_shape=",parmas_shape)
#parmas_truth = pd.read_csv(
#    #'synth_data/Xy_N=10000000_Epoch_N=1000_Nt=10_p=100/Beta_t.csv', 
#     'synth_data/Xy_N=' + args.Xy_N + 
#    '_Epoch_N=' + args.Epoch_N +
#    '_Nt=' + args.Nt +
#    '_p=' + args.p + '/Beta_t.csv',
#    #low_memory=False, 
#    index_col=0
#).iloc[:parmas_shape[0],:parmas_shape[1]]
#print("parmas_truth.shape=", parmas_truth.shape)
##print(parmas_truth.head())
#print("type(shard_data['predictors'])=", type(shard_data['predictors']))
#print("shard_data['predictors']=", shard_data['predictors'])
#embarrassingly_parallel.plot_CMC_parameter_path_(
#    parameter_history_obj.bo_list_history,
#    shard_data['predictors'],
#    parmas_truth
#)
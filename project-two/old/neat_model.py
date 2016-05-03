from neat import nn, population, statistics, visualize
pop = population.Population('neatConfig.txt')
pop.config.input_nodes = X.shape[1]
pop.config.output_nodes = 1

def get_predictions_from_neat(net, X):
    preds = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        preds[i] = net.serial_activate(X[i,:])[0]
    return preds

def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)
        data_test['pred_rel'] = get_predictions_from_neat(net, X_test)
        grouped = data_test.groupby('srch_id')
        g.fitness = grouped.apply(lambda x: ndcg_of_table_chunk(x)).mean()
        
pop.run(eval_fitness, 10)
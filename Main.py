import Preprocessing as Pre
import Feature_maker as fmaker
import Perceptron as P
import MST
import numpy as np
from scipy.sparse import csr_matrix
import time
if __name__ == '__main__':
    train_dict = Pre.get_file_as_dict("train")[0]
    f = fmaker.Feature_maker(train_dict)
    print(f.dimensions)
    f.init_all_features_indexes()

    print(f.dimensions)
    begin = time.time()
    weights = csr_matrix((1, f.dimensions))


    graph_with_weights = f.create_weighted_graph_for_sentence(1,weights.transpose())

    print("it took ",time.time()-begin)
    print("graph with weights:")
    print(graph_with_weights)
    value = MST.mst(0, graph_with_weights)
    print (value)

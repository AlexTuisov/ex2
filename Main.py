import Preprocessing as Pre
import Feature_maker as fmaker
import Perceptron as P
import MST
import numpy as np
from scipy.sparse import csr_matrix
import time
if __name__ == '__main__':
    train_dict, golden_standard = Pre.get_file_as_dict("test")
    f = fmaker.Feature_maker(train_dict,golden_standard)
    init_start = time.time()
    f.init_all_features_indexes()
    print("initialization of features took ",time.time()-init_start)
    p = P.Perceptron(f,20)
    p.run()
    test_set = Pre.get_file_as_dict("test")[0]


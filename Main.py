import Preprocessing as Pre
import Feature_maker as fmaker
import Perceptron as P
from multiprocessing import Pool

import time

def accuracy_measure(num_of_iterations, feature_maker):
    p = P.Perceptron(feature_maker, num_of_iterations)
    p.run()
    test_set, golden = Pre.get_file_as_dict("test")
    return (p.inference(test_set, False, golden))

if __name__ == '__main__':
    with Pool(4) as my_pool:
        train_dict, golden_standard = Pre.get_file_as_dict("train") #some
        f = fmaker.Feature_maker(train_dict,golden_standard,True)
        init_start = time.time()
        f.init_all_features_indexes()
        print("initialization of features took ", time.time()-init_start)
        iterations = []
        makers = [f]*1
        input_pool = zip(iterations, makers)
        list_of_accuracies = my_pool.starmap(accuracy_measure, input_pool)
        print (list_of_accuracies)





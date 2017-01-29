import Preprocessing as Pre
import Feature_maker as fmaker
import Perceptron as P
from multiprocessing import Pool

import time

def accuracy_measure(num_of_iterations, feature_maker):
    p = P.Perceptron(feature_maker, num_of_iterations)
    p.run()
    test_set, golden = Pre.get_file_as_dict("test")
    return (p.inference(test_set, True, golden))

if __name__ == '__main__':
    train_dict, golden_standard = Pre.get_file_as_dict("train")
    f = fmaker.Feature_maker(train_dict, golden_standard, True)
    init_start = time.time()
    f.init_all_features_indexes()
    print("initialization of features took ", time.time()-init_start)
    p = P.Perceptron(f, 20)
    p.run()
    test_set, golden = Pre.get_file_as_dict("competition")
    results_of_competition = p.inference(test_set, True, golden)
    Pre.print_the_results(results_of_competition)



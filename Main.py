import Preprocessing as Pre
import Feature_maker as fmaker
import Perceptron as P

import time
if __name__ == '__main__':
    train_dict, golden_standard = Pre.get_file_as_dict("train") #some
    f = fmaker.Feature_maker(train_dict,golden_standard,True)
    init_start = time.time()
    f.init_all_features_indexes()
    print("initialization of features took ",time.time()-init_start)
    iterations =[20]
    for iteration in iterations:
        p = P.Perceptron(f, iteration)
        p.run()
        test_set,golden = Pre.get_file_as_dict("test")
        p.inference(test_set,False,golden)

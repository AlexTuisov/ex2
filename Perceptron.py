

import scipy as sc
import mstk as mst
import Feature_maker
class Perceptron:

    def __init__(self,train_data,feature_maker):
        self.train_data = train_data
        self.feature_maker = feature_maker
        self.weights = sc.sparse.csc_matrix((1,self.feature_maker.dimensions))



    def run(self):
        print("")



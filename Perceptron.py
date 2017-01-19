

import scipy as sc
import mst as mst
import Feature_maker

class Perceptron:

    def __init__(self,feature_maker):
        self.feature_maker = feature_maker
        self.weights = sc.sparse.csc_matrix((1,self.feature_maker.dimensions))


    def convert_to_graph_weights(self,sentence_index):
        graph_no_weights = self.feature_maker.local_feature_dictionary[sentence_index]
        graph_with_weights ={}
        for parent in graph_no_weights:
            graph_with_weights[parent]={}
            for child in graph_no_weights:
                if parent!=child:
                    graph_with_weights[parent][child]=self.weights.dot(graph_no_weights[parent][child])

        return graph_with_weights

    def run(self):
        print("")



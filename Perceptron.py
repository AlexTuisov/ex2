

from scipy.sparse import csr_matrix
import mst as mst
import Feature_maker

class Perceptron:

    def __init__(self,feature_maker,global_iterations):
        self.feature_maker = feature_maker
        self.weights = csr_matrix((1,self.feature_maker.dimensions))
        self.global_iterations =global_iterations


    def convert_to_graph_weights(self,sentence_index):
        graph_no_weights = self.feature_maker.local_feature_dictionary[sentence_index]
        graph_with_weights ={}
        for parent in graph_no_weights:
            graph_with_weights[parent]={}
            for child in graph_no_weights:
                if parent!=child:
                    graph_with_weights[parent][child]=self.weights.dot(graph_no_weights[parent][child])

        return graph_with_weights



    def compare_trees(self):
        print("")


    def run(self):
        for iteration in xrange(0,self.global_iterations):
            print("global iteration number: ",iteration)
            for sentence in self.feature_maker.train_data:
                graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence,self.weights)
                maximum_spanning_tree=mst.mst(0,graph_with_all_weights)






from scipy.sparse import csr_matrix
import MST as mst
import time


class Perceptron:

    def __init__(self,feature_maker,global_iterations):
        self.feature_maker = feature_maker
        self.weights ={}
         #csr_matrix((1,self.feature_maker.dimensions),dtype=int)
        self.global_iterations = global_iterations

    def init_weights(self):
        for dimension in range(0,self.feature_maker.dimensions):
            self.weights[dimension] = 0


    def subtract(self,mst_vec):
        for index in mst_vec:
            self.weights[index] -= 1

    def add(self, real_vec):
        for index in real_vec:
            self.weights[index] += 1

    def compare_trees(self, mst_tree, true_tree):
        for parent in mst_tree:
            to_check = set(mst_tree[parent])
            if to_check != set(true_tree[parent]):
                return False
        return True

    def run(self):
        self.init_weights()
        for iteration in range(0, self.global_iterations):
            print("global iteration number: ", iteration+1)
            begin = time.time()
            for sentence in self.feature_maker.train_data:
                print ("working on sentence number ",sentence)
                graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence, self.weights)
                maximum_spanning_tree = mst.mst(0, graph_with_all_weights)
                golden_standard = self.feature_maker.golden_standard[sentence]
                if not self.compare_trees(maximum_spanning_tree, golden_standard):
                    """self.weights += (self.feature_maker.sentence_feature_dictionary[sentence] -
                                    self.feature_maker.create_feature_vector_from_tree(sentence, maximum_spanning_tree))"""
                    self.add(self.feature_maker.sentence_feature_dictionary[sentence])
                    self.subtract(self.feature_maker.create_feature_vector_from_tree(sentence,maximum_spanning_tree))

                if sentence%100 == 0:
                    print("took ",time.time()-begin)
        return self.weights


"""def convert_to_graph_weights(self,sentence_index):
        graph_no_weights = self.feature_maker.local_feature_dictionary[sentence_index]
        graph_with_weights = {}
        for parent in graph_no_weights:
            graph_with_weights[parent] = {}
            for child in graph_no_weights:
                if parent != child:
                    graph_with_weights[parent][child] = self.weights.dot(graph_no_weights[parent][child])
        return graph_with_weights"""
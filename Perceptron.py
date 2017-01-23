

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
            to_check = set(mst_tree[parent].keys())
            if to_check != set(true_tree[parent]):
                return False
        return True

    def number_of_correct(self,mst_tree,true_tree):
        correct = 0
        for parent in mst_tree:
            correct +=  len(set(mst_tree[parent].keys()).intersection(set(true_tree[parent])))
        return correct

    def run(self):
        self.init_weights()
        begin = time.time()
        for iteration in range(0, self.global_iterations):
            #print("global iteration number: ", iteration+1)

            for sentence in self.feature_maker.train_data:
             #   print("working on sentence number ",sentence)
                graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence, self.weights,self.feature_maker.train_data)
                maximum_spanning_tree = mst.mst(0, graph_with_all_weights)
                golden_standard = self.feature_maker.golden_standard[sentence]
                if not self.compare_trees(maximum_spanning_tree, golden_standard):
                    self.add(self.feature_maker.sentence_feature_dictionary[sentence])
                    self.subtract(self.feature_maker.create_feature_vector_from_tree(sentence,maximum_spanning_tree))
        print("perceptron run took :",time.time()-begin," with ",self.global_iterations," iterations")
        return self.weights


    def inference(self,test_set,real_test,golden_set):
        correct = 0
        total = 0
        for sentence in test_set:
            graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence, self.weights,test_set)
            maximum_spanning_tree = mst.mst(0, graph_with_all_weights)
            if not real_test:
                golden_standard = golden_set[sentence]
                correct += self.number_of_correct(maximum_spanning_tree,golden_standard)
                total += len(test_set[sentence])
        accuracy = float(float(correct)/total)
        print("The final accuracy is ",accuracy," achieved with ",self.global_iterations," iterations")


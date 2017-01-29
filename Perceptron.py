

from scipy.sparse import csr_matrix
import MST as mst
import time
import networkx as nx
import random
class Perceptron:




    def __init__(self,feature_maker,global_iterations):
        self.feature_maker = feature_maker
        self.weights ={}
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


    def connect_tree(self,mst):
        parent_set = set()
        children_set = set()
        for parent in mst:
            for child in mst[parent]:
                children_set.add(child)
            parent_set.add(parent)
        no_father_set = parent_set.difference(children_set)
        if no_father_set:
            print("no orphans")
            roots_child = random.sample(no_father_set,1)[0]
            mst[0]={}
            mst[0][roots_child] = 0
        return mst


    def run(self):
        self.init_weights()
        begin = time.time()
        problem = 0
        for iteration in range(0, self.global_iterations):
            for sentence in self.feature_maker.train_data:
                graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence, self.weights,self.feature_maker.train_data)
                maximum_spanning_tree = mst.mst(0, graph_with_all_weights)
                """if 0 not in maximum_spanning_tree:
                    maximum_spanning_tree = self.connect_tree(maximum_spanning_tree)"""
                golden_standard = self.feature_maker.golden_standard[sentence]
                if not self.compare_trees(maximum_spanning_tree, golden_standard):
                    self.add(self.feature_maker.sentence_feature_dictionary[sentence])
                    self.subtract(self.feature_maker.create_feature_vector_from_tree(sentence,maximum_spanning_tree))
        print("perceptron run took :",time.time()-begin," with ",self.global_iterations," iterations")
        print("problem with " ,problem)
        return self.weights





    def inference(self,test_set,real_test,golden_set):
        correct = 0
        total = 0
        result_dict = {}
        for sentence in test_set:
            graph_with_all_weights = self.feature_maker.create_weighted_graph_for_sentence(sentence, self.weights,test_set)
            maximum_spanning_tree = mst.mst(0, graph_with_all_weights)
            if not real_test:
                golden_standard = golden_set[sentence]
                correct += self.number_of_correct(maximum_spanning_tree,golden_standard)
                total += (len(test_set[sentence])-1)
                accuracy = float(float(correct)/total)

            else:
                result_dict[sentence] = maximum_spanning_tree

        if not real_test:
            print("The final accuracy is ",accuracy," achieved with ",self.global_iterations," iterations")
            return accuracy
        else:
            return result_dict
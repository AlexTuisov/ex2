from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from numba import jit
class Feature_maker:

    def __init__(self,train_data,golden_standard):
        self.train_data=train_data
        self.dimensions = 0
        self.feature_index = {}
        self.reverese_feature_index = {}
        self.special_delimiter = "@@<<<>>>@@"
        self.local_feature_dictionary={}
        self.sentence_feature_dictionary={}
        self.golden_standard =golden_standard

    def modify_feature_index(self,feature):
        if feature not in self.feature_index:
            self.feature_index[feature] = self.dimensions
            self.reverese_feature_index[self.dimensions] = feature
            self.dimensions += 1


    def get_relevant_features_basic(self,pword,ppos,cword,cpos):
        relevant_features = []
        unigram_pword_ppos = "<<head>>" + pword + self.special_delimiter + ppos
        relevant_features.append(unigram_pword_ppos)
        unigram_pword = "<<head>>" + pword
        relevant_features.append(unigram_pword)
        unigram_ppos = "<<head>>" + ppos
        relevant_features.append(unigram_ppos)

        unigram_cword_cpos = "<<child>>" + cword + self.special_delimiter + cpos
        relevant_features.append(unigram_cword_cpos)

        unigram_cword = "<<child>>" + cword
        relevant_features.append(unigram_cword)
        unigram_cpos = "<<child>>"+cpos
        relevant_features.append(unigram_cpos)

        bigram_ppos_cword_cpos =  "<<head>>" + ppos + self.special_delimiter + unigram_cword_cpos
        relevant_features.append(bigram_ppos_cword_cpos)
        bigram_pword_ppos_cword = unigram_pword_ppos + self.special_delimiter + "<<child>>" + cword
        relevant_features.append(bigram_pword_ppos_cword)
        bigram_ppos_cpos =  "<<head>>" + ppos + self.special_delimiter + "<<child>>"+cpos
        relevant_features.append(bigram_ppos_cpos)
        return relevant_features


    def add_features_basic_model(self,word_data,sentence):
        ppos = ""
        pword = ""
        if word_data['token head']!= -1:
            pword =sentence[word_data['token head']]['token']
            ppos =sentence[word_data['token head']]['token pos']

        cword =word_data['token']
        cpos = word_data['token pos']
        relevant_features = self.get_relevant_features_basic(pword,ppos,cword,cpos)
        for feature in relevant_features:
            self.modify_feature_index(feature)


    def init_all_features_indexes(self):
        for index in self.train_data:
            sentence = self.train_data[index]
            for word in sentence:
                word_data = sentence[word]
                self.add_features_basic_model(word_data,sentence)
        print("finished feature index creation")
        self.create_feature_vectors_for_all_training_sentences()
        print("finished creating feature vector for all sentences")


    def create_feature_vector_from_tree(self,sentence_index,graph):
        vector = csr_matrix((1,self.dimensions),dtype=int)
        for parent in graph:
            for child in graph[parent]:
                pword = self.train_data[sentence_index][parent]['token']
                ppos = self.train_data[sentence_index][parent]['token pos']
                cword= self.train_data[sentence_index][child]['token']
                cpos = self.train_data[sentence_index][child]['token pos']
                vector+=self.create_local_feature_vector(pword,ppos,cword,cpos)
        return vector



    def create_local_feature_vector(self,pword,ppos,cword,cpos):
        vector = lil_matrix((1,self.dimensions),dtype=int)
        relevant_features = self.get_relevant_features_basic(pword,ppos,cword,cpos)
        for feature in relevant_features:
            if self.feature_index.get(feature,False):
                index_of_feature =self.feature_index[feature]
                vector[0,index_of_feature] = 1
        return csr_matrix(vector)


    """def create_golden_standard(self):
        trees={}
        for sentence in self.train_data:
            trees[sentence]={}
            for word_data in self.train_data[sentence]:
                trees[sentence][word_data]={}
                for child in self.train_data[sentence][word_data]["token child"]:
                    trees[sentence][word_data][child]=0
        self.golden_standard = trees"""

    def create_feature_vectors_for_all_training_sentences(self):
        dictionary = {}
        for index in self.train_data:
            dictionary[index] = csr_matrix((1,self.dimensions))
            sentence = self.train_data[index]

            for word_index in sentence:
                pword = ""
                ppos = ""
                if sentence[word_index]['token head'] != -1:
                    pword = sentence[sentence[word_index]['token head']]['token']
                    ppos= sentence[sentence[word_index]['token head']]['token pos']
                cword = sentence[word_index]['token']
                cpos = sentence[word_index]['token pos']
                local_feature_vector = self.create_local_feature_vector(pword,ppos,cword,cpos)
                dictionary[index] += local_feature_vector
        self.sentence_feature_dictionary = dictionary



    def create_weighted_graph_for_sentence(self,sentence_index,weights):
        local_feature_dictionary = {}
        sentence = self.train_data[sentence_index]
        for word_index in sentence:
            local_feature_dictionary[word_index]={}
            for word_index1 in sentence:
                if word_index != word_index1:
                    pword = sentence[word_index]['token']
                    ppos= sentence[word_index]['token pos']
                    cword = sentence[word_index1]['token']
                    cpos = sentence[word_index1]['token pos']
                    local_feature_dictionary[word_index][word_index1] = -(self.create_local_feature_vector(pword,ppos,cword,cpos).dot(weights)[0, 0])
        return local_feature_dictionary


"""for child in word_data['token child']:
    unigram_cword_cpos = "<<child>>" + sentence[child]['token'] + self.special_delimiter + sentence[child]['token pos']
    relevant_features.append(unigram_cword_cpos)
    unigram_cword = "<<child>>" + sentence[child]['token']
    relevant_features.append(unigram_cword)
    unigram_cpos = "<<child>>" + sentence[child]['token pos']
    relevant_features.append(unigram_cpos)
    bigram_ppos_cword_cpos = unigram_ppos + self.special_delimiter + unigram_cword_cpos
    relevant_features.append(bigram_ppos_cword_cpos)
    bigram_pword_ppos_cword = unigram_pword_ppos + self.special_delimiter + unigram_cword
    relevant_features.append(bigram_pword_ppos_cword)
    bigram_ppos_cpos = unigram_ppos + self.special_delimiter + unigram_cpos
    relevant_features.append(bigram_ppos_cpos)"""

import math

class Feature_maker:

    def __init__(self,train_data,golden_standard,extended):
        self.train_data=train_data
        self.dimensions = 0
        self.feature_index = {}
        self.reverese_feature_index = {}
        self.special_delimiter = "@@<<<>>>@@"
        self.local_feature_dictionary={}
        self.sentence_feature_dictionary={}
        self.golden_standard =golden_standard
        self.extended = extended

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

        bigram_ppos_cword_cpos = "<<head>>" + ppos + self.special_delimiter + unigram_cword_cpos
        relevant_features.append(bigram_ppos_cword_cpos)
        bigram_pword_ppos_cword = unigram_pword_ppos + self.special_delimiter + "<<child>>" + cword
        relevant_features.append(bigram_pword_ppos_cword)
        #bigram_ppos_cpos =  "<<head>>" + ppos + self.special_delimiter + "<<child>>"+cpos
        #relevant_features.append(bigram_ppos_cpos)
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

    def add_features_extended_model(self,word_data,sentence,cindex):
        ppos = ""
        pword = ""
        if word_data['token head'] != -1:
            pword = sentence[word_data['token head']]['token']
            ppos = sentence[word_data['token head']]['token pos']
        cword = word_data['token']
        cpos = word_data['token pos']
        relevant_features = self.get_features_extended_model(pword, ppos, cword, cpos,word_data['token head'],cindex,sentence)
        for feature in relevant_features:
            self.modify_feature_index(feature)


    def init_all_features_indexes(self):
        for index in self.train_data:
            sentence = self.train_data[index]
            for word in sentence:
                word_data = sentence[word]
                self.add_features_basic_model(word_data,sentence)
                if self.extended:
                    self.add_features_extended_model(word_data,sentence,word)
        print("finished feature index creation")
        self.create_feature_vectors_for_all_training_sentences()
        print("finished creating feature vector for all sentences")


    def create_feature_vector_from_tree(self,sentence_index,graph):
        feature_vector =[]
        for parent in graph:
            for child in graph[parent]:
                pword = self.train_data[sentence_index][parent]['token']
                ppos = self.train_data[sentence_index][parent]['token pos']
                cword= self.train_data[sentence_index][child]['token']
                cpos = self.train_data[sentence_index][child]['token pos']
                feature_vector.extend(self.create_local_feature_vector(pword,ppos,cword,cpos,parent,child, self.train_data[sentence_index]))
        return feature_vector

    def create_local_feature_vector(self,pword,ppos,cword,cpos,parent,child, sentence):
        relevant_features = self.get_relevant_features_basic(pword,ppos,cword,cpos)
        relevant_indexes = []
        for feature in relevant_features:
            if self.feature_index.get(feature,False):
                index_of_feature =self.feature_index[feature]
                relevant_indexes.append(index_of_feature)
        if self.extended:
            relevant_features_extended = self.get_features_extended_model(pword, ppos ,cword,cpos,parent,child,sentence)
            for feature in relevant_features_extended:
                if self.feature_index.get(feature, False):
                    index_of_feature = self.feature_index[feature]
                    relevant_indexes.append(index_of_feature)
        return relevant_indexes


    def get_features_extended_model(self, pword, ppos, cword, cpos, parent_index, child_index, sentence):
        extended_features = []
        ppos_cpos_length = ppos+self.special_delimiter+cpos+self.special_delimiter+str(abs(parent_index-child_index))
        extended_features.append(ppos_cpos_length)
        pword_cword_length = pword+self.special_delimiter+cword+self.special_delimiter+str(abs(parent_index-child_index))
        extended_features.append(pword_cword_length)
        for word_index in range(parent_index+1,child_index-1):
            word_data = sentence[word_index]
            bpos = word_data['token pos']
            in_between_feature = ppos+self.special_delimiter+bpos+self.special_delimiter+cpos
            extended_features.append(in_between_feature)
        if (parent_index + 1) < len(sentence) and (child_index-1) > -1:
            ppos_pposplusone_cposminusone_cpos =self.special_delimiter.join((ppos,sentence[parent_index+1]['token pos'],sentence[child_index-1]['token pos'],cpos))
            extended_features.append(ppos_pposplusone_cposminusone_cpos)
        if (parent_index-1) > -1 and (child_index-1) > -1:
            pposminusone_ppos_cposminusone_cpos =self.special_delimiter.join((sentence[parent_index-1]['token pos'],ppos,sentence[child_index-1]['token pos'],cpos))
            extended_features.append(pposminusone_ppos_cposminusone_cpos)
        if (parent_index+1) < len(sentence) and (child_index+1) < len(sentence):
            ppos_pposplusone_cpos_cpospulsone = self.special_delimiter.join((ppos,sentence[parent_index+1]['token pos'],cpos,sentence[child_index+1]['token pos']))
            extended_features.append(ppos_pposplusone_cpos_cpospulsone)
        if (parent_index-1) > -1 and (child_index+1) < len(sentence):
            pposminusone_ppos_cpos_cposplusone = self.special_delimiter.join((sentence[parent_index-1]['token pos'],ppos,cpos,sentence[child_index+1]['token pos']))
            extended_features.append(pposminusone_ppos_cpos_cposplusone)
        return extended_features


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
            dictionary[index] = []#csr_matrix((1,self.dimensions))
            sentence = self.train_data[index]

            for word_index in sentence:
                pword = ""
                ppos = ""
                if sentence[word_index]['token head'] != -1:
                    pword = sentence[sentence[word_index]['token head']]['token']
                    ppos= sentence[sentence[word_index]['token head']]['token pos']
                cword = sentence[word_index]['token']
                cpos = sentence[word_index]['token pos']
                local_feature_vector = self.create_local_feature_vector(pword,ppos,cword,cpos,sentence[word_index]['token head'],word_index, sentence)
                dictionary[index].extend(local_feature_vector)
        self.sentence_feature_dictionary = dictionary

    def multiply_vectors(self,indexes,weights):
        result = 0
        for index in indexes:
            result += weights[index]
        return result

    def create_weighted_graph_for_sentence(self, sentence_index, weights, set):
        local_feature_dictionary = {}
        sentence = set[sentence_index]
        for word_index in sentence:
            local_feature_dictionary[word_index]={}
            for word_index1 in sentence:
                if word_index != word_index1:
                    pword = sentence[word_index]['token']
                    ppos= sentence[word_index]['token pos']
                    cword = sentence[word_index1]['token']
                    cpos = sentence[word_index1]['token pos']
                    local_feature_dictionary[word_index][word_index1] = -(self.multiply_vectors(self.create_local_feature_vector(pword,ppos,cword,cpos,word_index,word_index1, sentence),weights))
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
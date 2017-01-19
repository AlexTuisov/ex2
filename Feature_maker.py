import scipy as sc

class Feature_maker:

    def __init__(self,train_data,special_delimiter):
        self.train_data=train_data
        self.dimensions = 0
        self.feature_index = {}
        self.reverese_feature_index = {}
        self.special_delimiter = special_delimiter

    def modify_feature_index(self,feature):
        if feature not in self.feature_index:
            self.feature_index[feature] = self.dimensions
            self.reverese_feature_index[self.dimensions] = feature
            self.dimensions += 1

    def get_relevant_features_basic(self,pword,ppos,cword,cpos,sentence):
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
        unigram_cpos = "<<child>>"+cword
        relevant_features.append(unigram_cpos)
        bigram_ppos_cword_cpos = unigram_ppos + self.special_delimiter + unigram_cword_cpos
        relevant_features.append(bigram_ppos_cword_cpos)
        bigram_pword_ppos_cword = unigram_pword_ppos + self.special_delimiter + unigram_cword
        relevant_features.append(bigram_pword_ppos_cword)
        bigram_ppos_cpos = unigram_ppos + self.special_delimiter + unigram_cpos
        relevant_features.append(bigram_ppos_cpos)



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
        return relevant_features


    def add_features_basic_model(self,word_data,sentence):
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


    def create_local_feature_vector(self,pword,ppos,cword,cpos):
        vector = sc.sparse.lil_matrix((1,self.dimensions))
        relevant_features = self.get_relevant_features_basic(pword,ppos,cword,cpos)
        for feature in relevant_features:
            vector[1,self.feature_index[feature]]=1
        return sc.sparse.csc_matrix(vector)


    def create_feature_dictionary_for_sentences(self):
        local_feature_dictionary = {}
        for index in self.train_data:
            sentence = self.train_data[index]
            local_feature_dictionary[index] = {}
            for word_index in sentence:
                for word_index1 in sentence:
                    if word_index!=word_index1:
                        pword = sentence[word_index]['token']
                        ppos= sentence[word_index]['token pos']
                        cword = sentence[word_index1]['token']
                        cpos = sentence[word_index1]['token pos']
                        local_feature_dictionary[index][word_index][word_index1] = self.create_local_feature_vector(pword,ppos,cword,cpos)






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

    def get_relevant_features(self,word_data,sentence):
        relevant_features = []
        unigram_pword_ppos = "<<head>>" + sentence[word_data['token head']]['token'] + self.special_delimiter + sentence[word_data['token head']]['token pos']
        relevant_features.append(unigram_pword_ppos)
        unigram_pword = "<<head>>" + sentence[word_data['token head']]['token']
        relevant_features.append(unigram_pword)
        unigram_ppos = "<<head>>" + sentence[word_data['token head']]['token pos']
        relevant_features.append(unigram_ppos)

        unigram_cword_cpos = "<<child>>" + word_data['token'] + self.special_delimiter + word_data['token pos']
        relevant_features.append(unigram_cword_cpos)
        unigram_cword = "<<child>>" + "<<child>>" + word_data['token']
        relevant_features.append(unigram_cword)
        unigram_cpos = "<<child>>" + word_data['token pos']
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
        relevant_features = self.get_relevant_features(word_data,sentence)
        for feature in relevant_features:
            self.modify_feature_index(feature)

    def init_all_features(self):
        for index in self.train_data:
            sentence = self.train_data[index]
            for word in sentence:
                word_data = sentence[word]
                self.add_features_basic_model(word_data,sentence)








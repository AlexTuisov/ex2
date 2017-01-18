import Preprocessing as Pre
# import Feature_maker as fmaker
if __name__ == '__main__':
    train_dict = Pre.get_file_as_dict("train")
    # f = fmaker.Feature_maker(train_dict)
    # f.init_all_features()

    print (train_dict.popitem())

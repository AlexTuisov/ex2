import os


def get_path(type_of_file):
    path = os.path.dirname(__file__)
    print "____"
    print path
    if type_of_file == "test":
        absolute_path = os.path.join(path, "data/test.labeled")
    elif type_of_file == "train":
        absolute_path = os.path.join(path, "data/train.labeled")
    elif type_of_file == "competition":
        absolute_path = os.path.join(path, "data/comp.unlabeled")
    elif type_of_file == "answers":
        absolute_path = os.path.join(path, "data/comp.labeled")
    else:
        raise ValueError("wrong file type requested")
    print absolute_path
    return absolute_path


def get_file_as_dict(type_of_file):
    formatted_file = {}
    if type_of_file == "train":
        with open(get_path(type_of_file)) as unformatted_train_set:
            is_a_new_sentence = False
            sentence_as_dictionary = {}
            numerator = 0
            for row in unformatted_train_set:
                if len(row) < 10:
                    formatted_file[numerator] = sentence_as_dictionary
                    is_a_new_sentence = True
                    for key, value in tuple(sentence_as_dictionary.items()):
                        num_of_father = value["token head"]
                        if not num_of_father:
                            continue
                        sentence_as_dictionary[num_of_father]["token child"].append(key)
                    continue
                if is_a_new_sentence:
                    sentence_as_dictionary = {}
                    numerator += 1
                    is_a_new_sentence = False
                split_row = row.split()
                sentence_as_dictionary[int(split_row[0])] = {"token": split_row[1],
                                                        "token pos": split_row[3],
                                                        "token head": int(split_row[6]),
                                                        "token child": []}
    return formatted_file


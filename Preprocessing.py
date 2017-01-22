import os


def get_path(type_of_file):
    path = os.path.dirname(__file__)
    print ("____")
    print (path)
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
    print (absolute_path)
    return absolute_path


def get_file_as_dict(type_of_file):
    formatted_file = {}
    formatted_graph = {}
    if type_of_file == "train":
        with open(get_path(type_of_file)) as unformatted_train_set:
            is_a_new_sentence = False
            sentence_as_dictionary = {}
            numerator = 0
            for row in unformatted_train_set:
                if len(row) < 10:
                    is_a_new_sentence = True
                    child_of_root = []
                    for key, value in tuple(sentence_as_dictionary.items()):
                        num_of_father = value["token head"]
                        if not num_of_father:
                            child_of_root.append(key)
                            continue
                        sentence_as_dictionary[num_of_father]["token child"].append(key)
                    sentence_as_graph = make_graph_for_sentence(sentence_as_dictionary)
                    sentence_as_dictionary[0] = {"token": "root", "token pos": "root", "token head": -1,
                                                 "token child": child_of_root}
                    formatted_file[numerator] = sentence_as_dictionary
                    formatted_graph[numerator] = sentence_as_graph
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
    return formatted_file, formatted_graph


def make_graph_for_sentence(sentence_as_dict):
    graph_as_dictionary = {}
    for key, value in tuple(sentence_as_dict.items()):
        head = value["token head"]
        if head in graph_as_dictionary:
            graph_as_dictionary[head][key] = 0
        else:
            graph_as_dictionary[head] = {}
            graph_as_dictionary[head][key] = 0
        if key not in graph_as_dictionary:
            graph_as_dictionary[key] = {}
    return graph_as_dictionary

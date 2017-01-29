import os
import random


def get_path(type_of_file):
    path = os.path.dirname(__file__)
    print("____")
    print(path)
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
    numerator = 0
    if type_of_file == "train" or type_of_file == "test":
        with open(get_path(type_of_file)) as unformatted_train_set:
            is_a_new_sentence = False
            sentence_as_dictionary = {}
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

    if type_of_file == "competition":
        with open(get_path(type_of_file)) as unformatted_competition_set:
            is_a_new_sentence = False
            sentence_as_dictionary = {}
            for row in unformatted_competition_set:
                if len(row) < 10:
                    is_a_new_sentence = True
                    sentence_as_dictionary[0] = {"token": "root", "token pos": "root"}
                    formatted_file[numerator] = sentence_as_dictionary
                if is_a_new_sentence:
                    sentence_as_dictionary = {}
                    numerator += 1
                    is_a_new_sentence = False
                split_row = row.split()
                if random.random() < 0.0001:
                    print(row)
                    print(split_row)
                if split_row:
                    sentence_as_dictionary[int(split_row[0])] = {"token": split_row[1],
                                                                 "token pos": split_row[3]}
        return formatted_file, None


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


def print_the_results(container_list):
    with open(get_path("competition")) as input_file:
        with open(get_path("answers"), "w") as output_file:
            numerator = 0
            mst_easy_lookup = None
            is_a_new_sentence = True
            for row in input_file:
                if len(row) < 10:
                    output_file.write(row)
                    is_a_new_sentence = True
                if is_a_new_sentence:
                    mst_easy_lookup = {y.keys()[0]: x for x, y in container_list[numerator].items()}
                    numerator += 1
                    is_a_new_sentence = False
                split_old_row = row.split()
                head = str(mst_easy_lookup[split_old_row[0]])
                split_old_row[6] = head
                to_write = "\t".join(split_old_row)
                output_file.write(to_write)






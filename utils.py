import json
from bunch import Bunch


def read_json(path):
    with open(path) as j:
        res = json.load(j)
    return res


def save_json(path, obj):
    with open(path, 'w') as j:
        json.dump(obj, j)


def make_dot_dict(dic):
    dot_dic = Bunch()
    for key, value in dic.items():
        if isinstance(value, dict):
            dot_dic = make_dot_dict(dict)
        else:
            dot_dic[key] = value
    return dot_dic


def read_txt(path):
    with open(path) as f:
        res = f.read()
    return res


def save_txt(path, obj):
    with open(path, 'w') as f:
        f.write(obj)
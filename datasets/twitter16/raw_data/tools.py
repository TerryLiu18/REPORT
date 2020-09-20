import os
import json


def check_dirs_or_file(path, isprint=True):
    """check files or dirs, create files and dirs if not exists"""
    filename = os.path.basename(path)
    if len(filename.split('.')) == 1:
        if os.path.isdir(path):
            if isprint:
                print("{} already exists".format(path))
            return True
        else:
            os.makedirs(path)
            return True
    elif len(filename.split('.')) == 2:
        if os.path.isfile(path):
            if isprint:
                print("{} already exists".format(path))
            return True
        else:
            return False


def iterable2txt(x, path):
    f = open(path, 'w', encoding='utf-8')
    f.write(str(x))
    f.close()
    print("write to {}".format(path))


def txt2iterable(path):
    f = open(path, 'r', encoding='utf-8')
    a = f.read()
    iterable = eval(a)
    f.close()
    return iterable


def save_dict_to_json(item, path, overwrite=True):
    if os.path.exists(path) and overwrite is False:
        print("{} already exists".format(path))
    else:
        try:
            item = json.dumps(item, indent=4)
            with open(path, "w", encoding='utf-8') as f:
                f.write(item)
                print("success write dict to json: {}".format(path))
        except Exception as e:
            print("write error==>", e)


def read_dict_from_json(path):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print("{} successfully loaded!".format(path))
                return data
    except Exception as e:
        print("read error==>", e)


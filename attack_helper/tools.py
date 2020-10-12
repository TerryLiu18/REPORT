import os
from datetime import datetime
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


def _convert_time(user_create_time):
    if user_create_time is None:
        raise ValueError('user_create_time is None')
    user_time = user_create_time.split(' ', 1)
    user_time_str = ' '.join(user_time[1:])
    user_created = datetime.strptime(user_time_str,'%b %d %H:%M:%S +0000 %Y')
    year, month, day, hour = user_created.year, user_created.month, user_created.day,user_created.hour
    time_str = str(year)+'-'+str(month)+'-'+str(day)+'-'+str(hour)
    return time_str


def save_dict_to_json(item, path, overwrite=True):
    if os.path.exists(path) and overwrite is False:
        print("{} already exists".format(path))
    else:
        try:
            item = json.dumps(item, indent=4, ensure_ascii=False)
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


if __name__ == '__main__':
    user_create_time = 'Sun Sep 26 01:11:15 +0000 2010'
    _ = _convert_time(user_create_time)
    print(_)
    a = {'a': 3, 'b':2, 'c': 5}
    save_path = os.path.join(os.getcwd(), 'try.json')
    save_dict_to_json(a, save_path)
    data = read_dict_from_json(save_path)
    print(data)

def read_file(root):
    info = []
    file = open(root, 'rt')
    while True:
        string = file.readline()
        if not string:
            break
        info.append(string)
    return info


def write_file(file_path, file_list1, file_list2=None):
    if file_list2 is None:
        with open(file_path, 'w') as file:
            for string in file_list1:
                print(string, file=file)
    else:
        if len(file_list1) == len(file_list2):
            with open(file_path, 'w') as file:
                for string1, string2 in zip(file_list1, file_list2):
                    print(string1 + ' ' + string2, file=file)
        else:
            return
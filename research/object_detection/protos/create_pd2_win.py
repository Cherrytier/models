import os
import subprocess

this_path = os.path.abspath(os.path.realpath(__file__))
print(this_path)
file_list = os.listdir(os.path.dirname(this_path))
print(file_list)
for proto_file in file_list:
    if proto_file.endswith('proto'):
        print(proto_file)
        subprocess.check_call(
            ['protoc', os.path.join('object_detection', 'protos', proto_file),
             '--python_out=.'])

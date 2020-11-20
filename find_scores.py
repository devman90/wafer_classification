import glob

file_paths = glob.glob('./*/result.txt')

for file_path in file_paths:
    with open(file_path) as f:
        print(file_path, f.readline(), end='')


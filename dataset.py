import os
import shutil
import random


def split_dataset(dir_name: str, train_dir: str, val_dir: str, test_dir: str):

    file_paths = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".java"):
                file_paths.append(os.path.join(root, file))
    count = len(file_paths)
    random.shuffle(file_paths)
    print(count)
    train_paths = file_paths[:int(count*0.8)]
    val_paths = file_paths[int(count*0.8):int(count*0.9)]
    test_paths = file_paths[int(count*0.9):count]
    mv2dir(train_paths, train_dir)
    mv2dir(val_paths, val_dir)
    mv2dir(test_paths, test_dir)


def mv2dir(src_dir_paths: list, dst_dir: str):
    if os.path.exists(dst_dir) is False:
        os.mkdir(dst_dir)
    for path in src_dir_paths:
        shutil.copy(path, dst_dir)


if __name__ == "__main__":
    split_dataset("data/normal", "data/my_train_dir/normal",
                  "data/my_val_dir/normal", "data/my_test_dir/normal")
    split_dataset("data/webshell", "data/my_train_dir/webshell",
                  "data/my_val_dir/webshell", "data/my_test_dir/webshell")

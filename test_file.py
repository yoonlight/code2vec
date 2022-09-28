from argparse import ArgumentParser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv


@dataclass
class TestFile:
    path: str
    name: str
    method: str
    predict: float
    time_elapse: float


class FileReader:
    test_files: List[TestFile] = []

    def __init__(self, csv_file_name, file_path) -> None:
        self.csv_file_name: str = csv_file_name
        self.file_path: str = file_path

    def read_files(self):
        for dirpath, _, filenames in os.walk(self.file_path):
            for filename in filenames:
                self.test_files.append(
                    TestFile(dirpath, filename, None, None, None))

    def save2csv(self):
        with open(self.csv_file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["path", "name", "method", "time_elapse", "y_pred"])
            for file in self.test_files:
                writer.writerow(
                    [file.path, file.name, file.method, file.time_elapse, file.predict])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    PATH = Path(args.path)
    TEST_PATH = PATH / "my_test_dir"
    FILE_NAME = PATH / "test_file.csv"

    reader = FileReader(FILE_NAME, TEST_PATH)
    reader.read_files()
    input_filenames = reader.test_files
    reader.save2csv()

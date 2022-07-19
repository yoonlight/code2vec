import os
from dataclasses import dataclass
from typing import List
import csv

from extractor import Extractor


@dataclass
class TestFile:
    path: str
    name: str
    method: str
    predict: float


class FileReader:
    test_files: List[TestFile] = []

    def __init__(self, csv_file_name, file_path) -> None:
        self.csv_file_name: str = csv_file_name
        self.file_path: str = file_path

    def read_files(self):
        for dirpath, _, filenames in os.walk(self.file_path):
            for filename in filenames:
                self.test_files.append(TestFile(dirpath, filename, None, None))

    def save2csv(self):
        with open(self.csv_file_name, "w") as f:
            writer = csv.writer(f)
            for file in self.test_files:
                writer.writerow([file.path, file.name, file.method, file.predict])

@dataclass
class Config:
    MAX_CONTEXTS = 200

if __name__ == "__main__":
    TEST_PATH = "data/my_test_dir"
    FILE_NAME = "data/test_file.csv"

    reader = FileReader(FILE_NAME, TEST_PATH)
    reader.read_files()
    input_filenames = reader.test_files
    # reader.save2csv()

    MAX_PATH_LENGTH = 8
    MAX_PATH_WIDTH = 2
    JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
    config = Config()
    path_extractor = Extractor(config,
                               jar_path=JAR_PATH,
                               max_path_length=MAX_PATH_LENGTH,
                               max_path_width=MAX_PATH_WIDTH)
    for file in input_filenames:
        try:
            file_path = os.path.join(file.path, file.name)
            predict_lines, hash_to_string_dict = path_extractor.extract_paths(
                file_path)
        except ValueError as e:
            print(e)
            continue
        print(predict_lines)
        # raw_prediction_results = model.predict(predict_lines)
        # print(raw_prediction_results)

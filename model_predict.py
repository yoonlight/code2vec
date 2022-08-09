from typing import List
from extractor import Extractor
from keras_model import Code2VecModel
import os
import numpy as np

from test_file import TestFile

MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class InteractivePredictor:

    def __init__(self, config, model: Code2VecModel):
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def predict(self, input_filenames: List[TestFile]) -> List[TestFile]:
        print('Starting interactive prediction...')
        results: List[TestFile] = []
        for idx, file in enumerate(input_filenames):
            try:
                file_path = os.path.join(file.path, file.name)

                predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(
                    file_path)
            except ValueError as e:
                print(e)
                continue
            raw_prediction_results: List[np.ndarray] = self.model.predict(
                predict_lines)
            method_num = len(predict_lines)
            for result in raw_prediction_results:
                predict_result = TestFile(file.path, file.name, method_num, np.squeeze(result.round()))
                results.append(predict_result)
                
        return results

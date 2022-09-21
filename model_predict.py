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
    def extract_paths(self, path):
        MAX_CONTEXTS = 200
        f = open(path, "r", encoding="utf-8", errors="ignore")
        output = f.readlines()
        result = []
        for i, line in enumerate(output):
            parts = line.rstrip().split(' ')
            method_name = parts[0]
            current_result_line_parts = [method_name]
            contexts = parts[1:]
            for context in contexts[:MAX_CONTEXTS]:
                context_parts = context.split(',')
                context_word1 = context_parts[0]
                context_path = context_parts[1]
                context_word2 = context_parts[2]
                current_result_line_parts += ['%s,%s,%s' % (context_word1, context_path, context_word2)]
            space_padding = ' ' * (MAX_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result

    def predict(self, input_filenames: List[TestFile]) -> List[TestFile]:
        print('Starting interactive prediction...')
        results: List[TestFile] = []
        file_results: List[TestFile] = []
        for idx, file in enumerate(input_filenames):
            file_path = os.path.join(file.path, file.name)
            lines = self.extract_paths(file_path)
            raw_prediction_results: List[np.ndarray] = self.model.predict(
                lines)
            method_num = len(lines)
            predict_file_result = None
            for result in raw_prediction_results:
                y_true = np.squeeze(result.round())
                predict_result = TestFile(file.path, file.name, method_num, y_true)
                results.append(predict_result)
                if y_true == 1:
                    predict_file_result = TestFile(file.path, file.name, method_num, y_true)
            if predict_file_result is None:
                predict_file_result = TestFile(file.path, file.name, method_num, 0)
            file_results.append(predict_file_result)
                
        return file_results, results

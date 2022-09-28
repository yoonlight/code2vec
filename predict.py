from pathlib import Path
from keras_model import Code2VecModel
from model_predict import InteractivePredictor
from config import Config
from test_file import FileReader
from datetime import datetime

if __name__ == '__main__':

    config = Config(set_defaults=True, load_from_args=True, verify=True)
    args = config.arguments_parser().parse_args()
    PATH = Path(args.path)
    TEST_PATH = PATH / "my_test_dir"
    FILE_NAME = PATH / \
        f"test_file_result_file_{int(datetime.now().timestamp())}.csv"
    FILE_METHOD_NAME = PATH / \
        f"test_method_result_file_{int(datetime.now().timestamp())}.csv"

    reader = FileReader(FILE_NAME, TEST_PATH)
    method_reader = FileReader(FILE_METHOD_NAME, TEST_PATH)

    reader.read_files()
    input_filenames = reader.test_files
    model = Code2VecModel(config)
    model.convert_tflite_model()
    if config.PREDICT:
        start_time = datetime.now().replace(microsecond=0)

        predictor = InteractivePredictor(config, model)
        reader.test_files, method_reader.test_files = predictor.predict(
            input_filenames)
        end_time = datetime.now().replace(microsecond=0)
        print(end_time-start_time)

    reader.save2csv()
    method_reader.save2csv()

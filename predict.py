from keras_model import Code2VecModel
from model_predict import InteractivePredictor
from config import Config
from test_file import FileReader
from datetime import datetime

if __name__ == '__main__':

    config = Config(set_defaults=True, load_from_args=True, verify=True)
    TEST_PATH = "data/my_test_dir"
    FILE_NAME = "data/test_file.csv"

    reader = FileReader(FILE_NAME, TEST_PATH)
    reader.read_files()
    input_filenames = reader.test_files
    model = Code2VecModel(config)

    if config.PREDICT:
        start_time = datetime.now().replace(microsecond=0)

        predictor = InteractivePredictor(config, model)
        reader.test_files = predictor.predict(input_filenames)
        end_time = datetime.now().replace(microsecond=0)
        print(end_time-start_time)

    reader.save2csv()

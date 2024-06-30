import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions



# output from predict script not null
# output from predict script is str data type
# the output is Y for an example data

#fixtures ---> functions before test function --> ensure single_prediction


@pytest.fixture
def single_predictions():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_predictions):
    assert single_predictions is not None
    
def test_single_pred_str_type(single_predictions):
    assert isinstance(single_predictions.get("predictions")[0],str)
    
def test_single_pred_validate(single_predictions):
    assert single_predictions.get("predictions")[0] == "Y"
    
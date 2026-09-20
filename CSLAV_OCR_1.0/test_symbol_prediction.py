import numpy
from reader import (
    decode_predictions,
    prediction_class_index,
    stored_prediction_class_index,
    symbol_text_from_prediction,
)


def test_stored_prediction_class_index():
    stored_a = """[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0.]]"""
    assert stored_prediction_class_index(stored_a) == 1


def test_symbol_text_from_softmax_prediction():
    predictions_list = decode_predictions('predictions.txt')
    softmax_result = numpy.zeros((1, 49), dtype=float)
    softmax_result[0, 1] = 0.95
    assert symbol_text_from_prediction(softmax_result, predictions_list) == 'а'


def test_old_string_comparison_would_fail():
    predictions_list = decode_predictions('predictions.txt')
    softmax_result = numpy.zeros((1, 49), dtype=float)
    softmax_result[0, 1] = 0.95
    stored = predictions_list[3][1]
    assert stored != str(softmax_result)
    assert symbol_text_from_prediction(softmax_result, predictions_list) == 'а'


if __name__ == '__main__':
    test_stored_prediction_class_index()
    test_symbol_text_from_softmax_prediction()
    test_old_string_comparison_would_fail()
    print('All tests passed.')

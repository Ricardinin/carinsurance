import functions as pf
import constants
import pandas as pd


# =========== scoring method =========

def predict(input_data, scipy_object_model):
    # replace features with null values
    for column in constants.WITH_NULL_FEATURES:
        input_data[column] = pf.replace_null(input_data, column)

    # replace every categorical with its corresponding numerical value
    for column in constants.CATEGORICAL_FEATURES:
        input_data[column] = pf.transform_categorical(input_data, column, constants.LABEL_MAPPINGS[column])

    # new column for duration of a call (callEnd - callStart) in seconds
    input_data[constants.CALL_DURATION] = input_data[constants.CALL_DURATION_FEATURES].apply(pf.get_call_duration, axis=1)

    # transform callStart from hh:mm:ss to hh
    input_data['CallStart'] = input_data['CallStart'].apply(lambda time_string: time_string[:2])

    # drop unnecessary features
    for column in constants.DROP_FEATURES:
        input_data.drop(columns=[column], inplace=True)

    # Collocate the target column (CarInsurance) as last column
    dfAux = input_data.pop(constants.TARGET_CATEGORY)
    input_data[constants.TARGET_CATEGORY] = dfAux

    # Custom scaling
    for k, v in constants.RATIO_SCALER.items():
        input_data[k] = pd.to_numeric(input_data[k])
        input_data[k] = input_data[k] / v

    # make predictions
    # predictions = pf.predict(input_data, constants.MODEL_PATH)
    predictions = pf.predict(input_data, scipy_object_model)

    return predictions


# ============ scoring test ==========================


if __name__ == '__main__':
    from math import sqrt
    import numpy as np

    from sklearn.metrics import mean_squared_error, r2_score

    import warnings

    warnings.simplefilter(action='ignore')

    # Load data
    data = pf.load_data(constants.DATA_FILE_PATH)
    X_train, X_test, y_train, y_test = pf.split_train_test(data, constants.TARGET_CATEGORY)

    # load model
    object_model = pf.load_model(constants.MODEL_PATH)

    pred = predict(X_test, object_model)

    # determine mse and rmse
    print('test mse: {}'.format(int(
        mean_squared_error(y_test, np.exp(pred)))))
    print('test rmse: {}'.format(int(
        sqrt(mean_squared_error(y_test, np.exp(pred))))))
    print('test r2: {}'.format(
        r2_score(y_test, np.exp(pred))))
    print()

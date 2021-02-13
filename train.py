import functions as pf
import constants
import warnings

warnings.simplefilter(action='ignore')

print("Started training")

# Load data
data = pf.load_data(constants.DATA_FILE_PATH)

# split data set
X_train, X_test, y_train, y_test = pf.split_train_test(data, constants.TARGET_CATEGORY)

# replace features with null values
for column in constants.WITH_NULL_FEATURES:
    X_train[column] = pf.replace_null(X_train, column)

# replace every catergorical with its corresponding numerical value
for column in constants.CATEGORICAL_FEATURES:
    X_train[column] = pf.transform_categorical(X_train, column, constants.LABEL_MAPPINGS[column])

# new column for duration of a call (callEnd - callStart) in seconds
X_train[constants.CALL_DURATION] = X_train[constants.CALL_DURATION_FEATURES].apply(pf.get_call_duration, axis=1)

# transform callStart from hh:mm:ss to hh
X_train['CallStart'] = X_train['CallStart'].apply(lambda time_string: time_string[:2])

# drop unnecessary features
for column in constants.DROP_FEATURES:
    X_train.drop(columns=[column], inplace=True)

# Collocate the target column (CarInsurance) as last column
dfAux = X_train.pop(constants.TARGET_CATEGORY)
X_train[constants.TARGET_CATEGORY] = dfAux

# Custom scaling
for k, v in constants.RATIO_SCALER.items():
    X_train[k] = X_train[k] / v

# training
pf.train_model(X_train, y_train, constants.MODEL_PATH)

print("End training")

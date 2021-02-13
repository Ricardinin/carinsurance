import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import joblib


#  -----------------------------------------------------------------------------
#  PREPROCESSING FUNCTIONS
#  -----------------------------------------------------------------------------

def load_data(df_path):
    return pd.read_csv(df_path)


def replace_null(df, feature, replacement="unknown"):
    return df[feature].fillna(replacement)


def transform_categorical(df, feature, mappings):
    # replaces strings by numbers using mappings dictionary
    return df[feature].map(mappings)


def split_train_test(df, target_category, test_size=0.30, random_state=101):
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target_category], test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def get_call_duration(cols):
    # Get the difference between call_end and call_start
    call_start = cols[0]
    call_end = cols[1]
    h1, m1, s1 = call_start.split(':')
    h2, m2, s2 = call_end.split(':')
    total_secs1 = int(h1) * 3600 + int(m1) * 60 + int(s1)
    total_secs2 = int(h2) * 3600 + int(m2) * 60 + int(s2)
    return total_secs2 - total_secs1


def get_hour(cols):
    time = cols[0]
    return time[:2]


def train_model(df, target, output_path):
    # initialise the model
    log_model = LogisticRegression(max_iter=400, solver='newton-cg')

    # train the model
    log_model.fit(df, target)

    # save the model
    joblib.dump(log_model, output_path)

    return None


# def predict(df, model_file_name):
#     model = joblib.load(model_file_name)
#     return model.predict(df)


# Functions to be called from rest api
def load_model(model_file_name):
    return joblib.load(model_file_name)


def predict(df, scipy_object_model):
    return scipy_object_model.predict(df)

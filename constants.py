#  -----------------------------------------------------------------------------
#  DATA FILE PATH
#  -----------------------------------------------------------------------------

DATA_FILE_PATH = "carInsurance_train.csv"
# DATA_FILE_PATH = "/home/osboxes/Downloads/Genesys/carInsurance_train.csv"
MODEL_PATH = "carInsurance_log_reg.pkl"
# MODEL_PATH = "/home/osboxes/Downloads/Genesys/carInsurance_log_reg.pkl"
TARGET_CATEGORY = "CarInsurance"

LABELS = {
    "Job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "Marital": ["divorced", "married", "single"],
    "Education": ["primary", "secondary", "tertiary", "unknown"],
    "Communication": ["cellular", "telephone", "unknown"],
    "LastContactMonth": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                         "oct", "nov", "dec"],
    "Outcome": ["failure", "other", "success", "unknown"]
}

LABEL_MAPPINGS = {
    "Job": {"admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3,
            "management": 4, "retired": 5, "self-employed": 6, "services": 7,
            "student": 8, "technician": 9, "unemployed": 10, "unknown": 11},
    "Marital": {"divorced": 0, "married": 1, "single": 2},
    "Education": {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3},
    "Communication": {"cellular": 0, "telephone": 1, "unknown": 2},
    "LastContactMonth": {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                         "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12},
    "Outcome": {"failure": 0, "other": 1, "success": 2, "unknown": 3}
}

#  -----------------------------------------------------------------------------
#  FEATURE ENGINEERING
#  -----------------------------------------------------------------------------

# features with null(NA) values
WITH_NULL_FEATURES = ["Job", "Education", "Communication", "Outcome"]

# features to map
CATEGORICAL_FEATURES = ["Job", "Marital", "Education", "Communication", "LastContactMonth", "Outcome"]

# features to calculate call_duration. Keep the order
CALL_DURATION_FEATURES = ['CallStart', 'CallEnd']
CALL_DURATION = "CallDuration"

# features to drop
DROP_FEATURES = ["CallEnd"]

# features to transform
TRANS_FEATURES = ["CallStart"]

# features to scale with ration
RATIO_SCALER = {
    "Balance": 100,
    "Age": 2,
    "DaysPassed": 5,
    "CallDuration": 15
}

from flask import Flask, jsonify, request
import pandas as pd
import functions as pf
import constants
import score

# load model
object_model = pf.load_model(constants.MODEL_PATH)

# app
app = Flask(__name__)


# routes
@app.route('/genesys/prediction', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = score.predict(data_df, object_model)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

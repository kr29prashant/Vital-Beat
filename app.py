#
from flask import Flask, render_template, request

import pickle

import numpy as np

app = Flask(__name__)

model = pickle.load(open('heartRandomForest.pkl', 'rb'))


@app.route('/')

def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    print(request.form)

    s = ()
    int_features = [float(x) for x in request.form.values()]

    final = np.array(int_features)
    new = final.reshape(1,-1)
    print(int_features)
    print(new.shape)
    prediction = model.predict(new)

    if prediction == 1:

        return render_template('index.html', prediction_text = "Die")
    else:
        return render_template('index.html',prediction_text = "Not Die")


if __name__ == '__main__':
    app.run(debug=True)




#
#
# from flask import Flask, render_template, request
# import pickle
# import numpy as np
#
# app = Flask(__name__)
#
# # Load the trained model
# model = pickle.load(open('heart_disease_model_Latest.pkl', 'rb'))
#
# # Load the scaler (if used in the preprocessing step)
# scaler = pickle.load(open('scaler.pkl', 'rb'))
#
# @app.route('/')
# def hello_world():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     print(request.form)
#
#     int_features = [float(x) for x in request.form.values()]
#     final_features = np.array(int_features).reshape(1, -1)
#
#     # Preprocess the input features if necessary
#     if scaler:
#         final_features = scaler.transform(final_features)
#
#     prediction = model.predict(final_features)
#
#     if prediction == 1:
#         return render_template('output.html')
#     else:
#         return render_template('output2.html')
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# app = Flask(__name__)
#
# model = pickle.load(open('heart_disease_model_Latest.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))
# column_names = pickle.load(open('column_names.pkl', 'rb'))
#
# @app.route('/')
# def hello_world():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     int_features = [float(x) for x in request.form.values()]
#     final_features = np.array(int_features).reshape(1, -1)
#     scaled_features = scaler.transform(final_features)
#     scaled_df = pd.DataFrame(scaled_features, columns=column_names)
#
#     print(scaled_df)
#
#     prediction = model.predict(scaled_df)
#
#     if prediction == 1:
#         return render_template('output.html')
#     else:
#         return render_template('output2.html')
#
# if __name__ == '__main__':
#     app.run(debug=True)



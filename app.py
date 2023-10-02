# pip install flask

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__, static_url_path='/static')


path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(path,'house_model_rfht.pkl'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    bed = int(request.form['bedrooms'])
    loc = int(request.form['location'])
    size = int(request.form['size'])
    status = int(request.form['status'])
    seller = int(request.form['seller'])
    Type = int(request.form['type'])

    # Create input data array
    input_data = np.array([[ loc, size, status, seller,bed, Type]])

    # Predict the price using the pre-trained model
    predicted_price = model.predict(input_data)[0]
    formatted_price = "{:.2f}".format(predicted_price)

    # Render the result page with predicted price
    return render_template('index.html', predicted_price = formatted_price)

if __name__ == '__main__':
    app.run()
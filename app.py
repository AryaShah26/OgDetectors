from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
import pickle

with open('model.pkl', 'rb') as file:
   model= pickle.load( file)
print(model)
print("Model saved successfully!")
@app.route('/')
def index():
    return render_template("index1.html")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    time = data.get('time')
    amount = data.get('amount')
    prediction = model.predict([[time, amount]])
    return jsonify({'fraud_prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)
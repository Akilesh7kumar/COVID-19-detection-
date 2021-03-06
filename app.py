from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def man():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
        app.run(debug=True,use_reloader=False)


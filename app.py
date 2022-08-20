from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output_as_exp = (prediction[0][0])
    output = f"{output_as_exp:.2f}"

    if (output_as_exp > 0.5):
        danger = f'danger! prob for fire is: {output}'
        return render_template('after_submit.html', result = danger)
    elif (output_as_exp <= 0.5):
        all_good = f'you are safe! prob for fire is: {output}'
        return render_template('after_submit.html', result = all_good)



if __name__ == '__main__':
    app.run(debug=True)
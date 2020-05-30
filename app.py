from flask import Flask,render_template,request
import pickle
import re
import string
import nltk
from sklearn import svm

app = Flask(__name__)
filename = 'nlp_models.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transforms.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data =[message] 
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)





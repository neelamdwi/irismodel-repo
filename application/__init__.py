from flask import Flask, request, Response, json
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# train model
X_train, X_test, y_train, y_test = train_test_split(X, y)
rfc = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
rfc.fit(X_train, y_train)

#create pickle file
filename = 'rfc_model'
pickle.dump(rfc, open(filename, 'wb'))

# Load the model from pickle
rfc_model = pickle.load(open(filename,'rb'))

#create api
@app.route('/api/', methods=['GET', 'POST'])
@app.route('/api',methods=['GET', 'POST'])
def predict():
    # Get the data from POST request
    data = request.get_json(force=True)
    requestData = [data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]
    requestData = np.array([requestData])

    # Make prediction using model 
    prediction = rfc_model.predict(requestData)
    return Response(json.dumps(int(prediction[0])))

if __name__ == '__main__':
   app.run()
    
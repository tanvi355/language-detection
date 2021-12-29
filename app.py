import pickle #to load the picked model
from flask_cors import CORS #for deployment phase

# load the model
model = pickle.load(open('finalized_model.sav', 'rb'))

# load the vectorizer
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

#This converts the input text string to lower case
def preprocess(text):
    text = text.lower()
    
    return text

# make a prediction using the model created
def pred(text):

    predicted=(model.predict(vectorizer.transform(text)))
    return str(predicted[0])

## making the routes, recive and send data using flask functionalities

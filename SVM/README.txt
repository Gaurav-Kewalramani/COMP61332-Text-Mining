The model is already downloaded and named as svm_model2.joblib along with vectorizer and encoder named as tfidf_vectorizer2.joblib and label_encoder2.joblib respectively


To use the model, simply drag and drop it on colab and copy past the following code : 


import joblib

# Load the model, vectorizer, and label encoder
svm_model = joblib.load('svm_model2.joblib')
vectorizer = joblib.load('tfidf_vectorizer2.joblib')
label_encoder = joblib.load('label_encoder2.joblib')

print("Model, vectorizer, and label encoder loaded successfully.")


import re

# Function to preprocess the input text 
def preprocess(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to predict relation from an input sentence
def predict_relation(sentence):
    # Preprocess the input sentence
    processed_text = preprocess(sentence)

    # Convert the processed text into TF-IDF features using the loaded vectorizer
    sentence_features = vectorizer.transform([processed_text])

    # Predict the relation using the loaded SVM model
    prediction = svm_model.predict(sentence_features)

    # Convert the predicted label back to its original relation name
    predicted_relation = label_encoder.inverse_transform(prediction)[0]

    return predicted_relation



# Example sentence to test the prediction
example_sentence = "Apple was founded by Gaurav"

# Predict the relation
predicted_relation = predict_relation(example_sentence)

# Display the result
print(f"Sentence: {example_sentence}")
print(f"Predicted Relation: {predicted_relation}")



The main code File is named as SVM_main where the main code for the SVM model is.

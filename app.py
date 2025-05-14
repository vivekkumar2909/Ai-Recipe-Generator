from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

app = Flask(__name__)

# Loading the saved model
model = tf.keras.models.load_model('recipe_generation_model.h5')

# Loading the dataset
data = pd.read_csv('dataset.csv')

data.drop_duplicates(subset=['TranslatedInstructions'], inplace=True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['TranslatedInstructions'])

# Maximum sequence length
max_sequence_length = 200

# Function to generate the recipe
def generate_recipe(seed_text, next_words=100):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Defining the home page of our site
@app.route('/')
def index():
    return render_template('index.html')

# Fetching the user input and displaying the generated recipe
@app.route('/generate_recipe', methods=['POST'])
def generate():
    user_input = request.form['user_input']
    generated_recipe = generate_recipe(user_input)
    return render_template('index.html', user_input=user_input, generated_recipe=generated_recipe)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
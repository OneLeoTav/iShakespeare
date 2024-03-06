
# Import libraries and dependencies
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import json
import random

# Import useful functions
from utils import unique_lemmatized_tokens
from utils import generate_dataset_bag_of_words
from utils import individual_bag_of_words
from utils import Shakespeare

import pytest

"""Import the dataset"""
with open('data/data.json', 'r') as json_file:
    data = json.load(json_file)

"""Call the trained Neural Network and instantiate the model"""

# Set up useful hyperperameters
input_dim, hidden_dim, output_dim = 165, 165, 14

# Instantiate the model
model = Shakespeare(input_dim = 165,
                    hidden_dim = 165,
                    output_dim = 14,
                    dropout = 0.2
)

# Load the trained PyTorch model
model.load_state_dict(torch.load("model/chatbot_model.pt"))

unique_words = []
for intent in data['intents']:
    unique_words += intent['patterns']

unique_words = unique_lemmatized_tokens(unique_words)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods = ["GET", "POST"])
def get_user_prompt():
    user_prompt = request.form["msg"]
    return get_ishakespeare_mind(user_prompt)


def get_ishakespeare_mind(query: str) -> str:
    model.eval()
    with torch.no_grad():
        # Convert the input sentence into a PyTorch tensor
        sentence_torch = torch.Tensor(individual_bag_of_words(query, unique_words))

        # Make a prediction using the trained model
        output = model(sentence_torch)
    
        # Get the predicted class and associated probability
        logit_max, predicted_class = torch.max(output, dim=0)
        soft = nn.Softmax(dim=0)
        predicted_prob = soft(logit_max).item()

        # Decode the predicted class into a string tag
        predicted_tag = predicted_class.item()
        decoded_string = intent_dict[predicted_tag]
    
        if predicted_prob >= 0.6:
            for intent in data['intents']:
                # Access the patterns within each intent
                if intent['tag'] == decoded_string:
                    possible_answers = intent['responses']
                    aleatoric_answer_idx = random.randint(0, len(possible_answers) - 1)
                    final_answer = possible_answers[aleatoric_answer_idx]
                    return final_answer
        else:
            return 'I am not sure I comprehend. May I ask you to rephrase your query ?'


if __name__ == '__main__':

    """Step 0: Deploy tests to ensure everything works properly"""
    pytest.main(['-v', 'test.py'])  

    """Step 1: Retrieve all unique words in the corpus"""
    unique_words = []
    for intent in data['intents']:
        unique_words += intent['patterns']

    unique_words = unique_lemmatized_tokens(unique_words)


    """Step 2:
        (i) retrieve all labels associated to each sentences in the corpus
        (ii) generate the intent_dict (key: int, value: pattern) to be used afterwards to retrieve
            the categoriy associated to the prediction of the ANN
        (iii) generate each question under TF format (dim[0] = num_questions, dim[1] = len(unique_words))
    """
    label, intent_dict, bag_of_words = generate_dataset_bag_of_words(data, unique_words)

    app.run(host='0.0.0.0', port=2424, debug=True)
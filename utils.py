from typing import List, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re


def tokenize_sentence(sentence: str) -> List[str]:
    """
    Tokenize a sentence (string) and returns a list of tokens (List[str])
    """
        
    return word_tokenize(sentence) 


def lemmatize_sentence(sentence: List[str]) -> List[str]:
    """
    Lemmatize a tokens (List[str]) and returns a list of lemme/root words (List[str])
    """    
    
    return [SnowballStemmer("english").stem(sentence[i]) for i in range(len(sentence))]


def remove_punctuation(strings: Union[str, List[str]]) -> List[str]: 
    """
    Removes punctuation marks from a list of strings.

    Args:
    - List[str]: A list of strings representing tokens.

    Returns:
    - List[str]: A new list of strings with punctuation marks removed.
    """
    
    # Define a regular expression pattern for matching punctuation
    punctuation_pattern = re.compile(r'[^\w\s]|_')

    # Apply the pattern to each string in the list and replace punctuation with an empty string
    cleaned_strings = [s for s in strings if not punctuation_pattern.search(s)]

    return cleaned_strings

def unique_lemmatized_tokens(all_intents: List[str]) -> List[str]:
    """
    Tokenizes, lemmatizes, and removes punctuation from a list of sentences,
    returning a list of unique lemmatized tokens in a given corpus.

    Args:
    - all_intents (List[str]): A list of sentences representing different intents.

    Returns:
    - List[str]: A list of unique lemmatized tokens extracted from the input sentences.
    """
    
    all_intents_tokenized = []
    for sentences in all_intents:
        tokenized = tokenize_sentence(sentences)
        lemmatized = lemmatize_sentence(tokenized)
        cleaned = remove_punctuation(lemmatized)
        all_intents_tokenized += cleaned
    lemmatized_tokens = list(set(all_intents_tokenized))
    lemmatized_tokens.sort()
    return lemmatized_tokens


def generate_dataset_bag_of_words(intents_list, unique_words):
    """Create a bag of words representation for the given intents list and unique words.

    Parameters:
    - intents_list (dict): A dictionary containing a list of intents, where each intent has a 'patterns' field.
    - unique_words (list): A list of unique words in the vocabulary.

    Returns:
    - label (list): A list of intent labels.
    - intent_dict (dict): A dictionary mapping intent indices to their corresponding tags.
    - bag_of_words (list): A list of bag of words representations for each pattern.
    """

    labels = []
    intent_dict = {}
    bag_of_words = []

    for (idx_intent, intent) in enumerate(intents_list['intents']):
        # Access the patterns within each intent
        patterns = intent['patterns']
        tag = intent['tag']
        intent_dict[idx_intent] = tag
        labels += [idx_intent] * len(patterns)

        for (i, sentence) in enumerate(patterns):
            tokenized = tokenize_sentence(sentence)
            lemmatized = lemmatize_sentence(tokenized)
            cleaned = remove_punctuation(lemmatized)

            # Initialize a list of zeros for the bag of words representation
            pattern_bag = [0] * len(unique_words)

            for word in cleaned:
                if word in unique_words:
                    j = unique_words.index(word)
                    pattern_bag[j] += 1

            # Append the bag of words representation to the list
            bag_of_words.append(pattern_bag)
            
    bag_of_words = np.array(bag_of_words)
        
    return labels, intent_dict, bag_of_words


def individual_bag_of_words(sentence, unique_words):
    """Create a bag of words representation for the given intents list and unique words.

    Parameters:
    - intents_list (dict): A dictionary containing a list of intents, where each intent has a 'patterns' field.
    - unique_words (list): A list of unique words in the vocabulary.

    Returns:
    - label (list): A list of intent labels.
    - intent_dict (dict): A dictionary mapping intent indices to their corresponding tags.
    - bag_of_words (list): A list of bag of words representations for each pattern.
    """

    tokenized = tokenize_sentence(sentence)
    lemmatized = lemmatize_sentence(tokenized)
    cleaned = remove_punctuation(lemmatized)

    # Initialize a list of zeros for the bag of words representation
    pattern_bag = [0] * len(unique_words)

    for word in cleaned:
        if word in unique_words:
            j = unique_words.index(word)
            pattern_bag[j] = 1

    individual_bag_of_words = np.array(pattern_bag)
        
    return individual_bag_of_words


def define_data_split(train: int, val: int, test: int):
    
    train_idx = range(0, train)
    val_idx = range(train, train+val)
    test_idx = range(train+val, train+val+test)
    
    return train_idx, val_idx, test_idx


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy of the predictions

    Args:
    - output (torch.Tensor): The tensor containing predictions.
    - labels (torch.Tensor): The tensor containing true labels.

    Returns:
    - float: The accuracy of the predictions.
    """

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def init_weights(m: nn.Module) -> None:
    """Initialize the weights of the model

    Args:
    - m (nn.Module): The neural network module for weight initialization.
    """

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)


# def classify_and_generate_response(model: nn.Module,
#                                    intents_list: dict,
#                                    unique_words: list,
#                                    sentence: str
# ) -> str:
    
#     """
#     Classify the given sentence using a trained PyTorch model and generate a response.

#     Parameters:
#     - model (torch.nn.Module): The trained PyTorch model for classification.
#     - intents_list (dict): A dictionary containing the intents and their patterns/responses.
#     - unique_words (list): A list of unique words used for encoding the sentence.
#     - sentence (str): The input sentence to be classified and responded to.

#     Returns:
#     None: Prints the generated response or a prompt for rephrasing.
#     """

#     # Set the model to evaluation mode and disable gradient computation
#     model.eval()
#     with torch.no_grad():
#         # Convert the input sentence into a PyTorch tensor
#         sentence_torch = torch.Tensor(individual_bag_of_words(sentence, unique_words))
        
#         # Make a prediction using the trained model
#         output = model(sentence_torch)

#     # Get the predicted class and associated probability
#     logit_max, predicted_class = torch.max(output, dim=0)
#     soft = nn.Softmax(dim=0)
#     predicted_prob = soft(logit_max).item()

#     # Decode the predicted class into a string tag
#     predicted_tag = predicted_class.item()
#     decoded_string = intent_dict[predicted_tag]

#     if predicted_prob >= 0.6:
#         for intent in intents_list['intents']:
#             # Access the patterns within each intent
#             if intent['tag'] == decoded_string:
#                 possible_answers = intent['responses']
#                 aleatoric_answer_idx = random.randint(0, len(possible_answers) - 1)
#                 final_answer = possible_answers[aleatoric_answer_idx]
#                 return str(final_answer)
#     else:
#         return 'Could you please rephrase your query'
        

class Shakespeare(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, dropout=None):
        super(Shakespeare, self).__init__()
        
        self.dropout = dropout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        
        self.linear_relu_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
            
        self.linear_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):

        x = self.linear_relu_input(X)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear_relu_hidden(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.linear_relu_hidden(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear_relu_hidden(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        
        x = self.linear_output(x)

        return x
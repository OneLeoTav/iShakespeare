import string
import numpy as np
from utils import tokenize_sentence
from utils import remove_punctuation
from utils import unique_lemmatized_tokens
from utils import generate_dataset_bag_of_words
from utils import individual_bag_of_words
from utils import define_data_split

from nltk.stem.snowball import SnowballStemmer

import pytest

def test_tokenize_sentence():

    """Description. Test for tokenize_sentence() function"""

    test_sentence = "She sings beautifully !"
    assert tokenize_sentence(test_sentence) == ['She', 'sings', 'beautifully', '!'], "Issue detcted with the tokenize_sentence function"


def test_lemmatize_sentence():
    """Description. Test for lemmatize_sentence() function"""
    
    tokenized_test_phrase = ['She', 'sings', 'beautifully', '!']
    
    assert [SnowballStemmer("english").stem(tokenized_test_phrase[i]) for i in range(len(tokenized_test_phrase))] == ['she', 'sing', 'beauti', '!'], "Issue detected with the tokenize_lemmatize function"


def test_remove_punctuation():

    """Description. Test for the remove_punctuation() function """

    all_punctuation = string.punctuation
    no_punctuation = remove_punctuation(all_punctuation)
    assert not no_punctuation, "The list should be empty after removing punctuation."


def test_unique_lemmatized_tokens():

    """Description. Test for the unique_lemmatized_tokens() function """

    all_intents = [
        "This is a sample sentence.",
        "Another sample sentence here.",
        "Yet another example for testing."
    ]

    expected_output = [
        'a',
        'anoth', 
        'exampl', 
        'for', 
        'here', 
        'is', 
        'sampl', 
        'sentenc', 
        'test', 
        'this', 
        'yet'
    ]

    result = unique_lemmatized_tokens(all_intents)

    assert result == expected_output, "Issue detected with the unique_lemmatized_tokens function"


def test_generate_dataset_bag_of_words():
    """Description. Test for the generate_dataset_bag_of_words() function"""

    intents_list = {
        'intents': [
            {'tag': 'greeting', 'patterns': ['Hello', 'Hi', 'Hey']},
            {'tag': 'farewell', 'patterns': ['Goodbye', 'See you', 'Hope to see you again soon']}
        ]
    }
    unique_words = ['hello', 'hi', 'hey', 'goodbye', 'see', 'you', 'hope', 'to', 'again', 'soon']

    expected_labels = [0, 0, 0, 1, 1, 1]
    expected_intent_dict = {0: 'greeting', 1: 'farewell'}
    expected_bag_of_words = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ])

    result_labels, result_intent_dict, result_bag_of_words = generate_dataset_bag_of_words(intents_list, unique_words)

    assert result_labels == expected_labels, 'Issue with the generation of the corpus bag of words'
    assert len(result_labels) == result_bag_of_words.shape[0] == expected_bag_of_words.shape[0], 'The number of labels do not match the number of patterns.'
    assert result_intent_dict == expected_intent_dict, 'Issue with generate_dataset_bag_of_words function'
    assert result_bag_of_words.all() == expected_bag_of_words.all(), 'Issue with generate_dataset_bag_of_words function'


def test_individual_bag_of_words():
    
    """Description. Test for the individual_bag_of_words() function"""

    unique_words = ['hello', 'hi', 'hey', 'goodbye', 'see', 'you', 'hope', 'to', 'again', 'soon']
    test_sentence = 'Hello there, how are you'
    expected_result = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ])
    
    result = individual_bag_of_words(test_sentence, unique_words)
    
    assert expected_result.all() == result.all(), "Issue with the individual bag of words representation."


def test_define_data_split():

    """Description. Test for the define_data_split function"""
   
    expected_bag_of_words = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ])
    
    train, val, test = 4, 1, 1
    _, __, test_idx = define_data_split(train, val, test)
    assert (test_idx[-1] + 1) == expected_bag_of_words.shape[0], 'Issue with the train/val/test split'
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

def clean_up(s):
    """
    Cleans up numbers, URLs, and special characters from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    # Remove URLs
    s = re.sub(r'http\S+', '', s)
    
    # Remove numbers and special characters
    s = re.sub(r'[^a-zA-Z\s]', '', s)
    
    return s.lower()

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    return word_tokenize(s)

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Stemming
    l = [stemmer.stem(word) for word in l]
    
    # Lemmatizing
    l = [lemmatizer.lemmatize(word) for word in l]
    
    return l

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in l if word not in stop_words]

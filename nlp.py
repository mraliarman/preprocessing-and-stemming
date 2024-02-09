import numpy as np
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# nltk.download('stopwords')
# nltk.download('punkt')

# Step 1: Opening file
def read_dataset(file_path):
    print('1')
    return pd.read_csv(file_path)

# Step 2: Handle missing values by replacing with the mean
def handle_missing_values(data):
    print('2')
    data_filled = data.fillna(data.mean())
    return data_filled

# Step 3: Text Preprocessing
def preprocess_text(text):
    print('3')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    text = ' '.join(word for word in tokens if word not in string.punctuation and not word.isdigit)
    return text

# Step 4: Remove Stop Words
def remove_stop_words(text):
    print('4')
    stop_words = set(stopwords.words('english'))  # Assuming you want to remove English stop words
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Step 5: Correct Spelling
def correct_spelling(text):
    print('5')
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

# Step 6: Stemming Words
def stem_words(text):
    print('6')
    stemmer = SnowballStemmer('english')  # Assuming you want to stem English words
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Main Execution
file_path = 'data.csv'
data = read_dataset(file_path)
data_filled = handle_missing_values(data)

# Apply additional processing steps
data_processed = data_filled.copy()
data_processed['TEXT'] = data_processed['TEXT'].apply(preprocess_text)
data_processed['TEXT'] = data_processed['TEXT'].apply(remove_stop_words)
data_processed['TEXT'] = data_processed['TEXT'].apply(correct_spelling)
data_processed['TEXT'] = data_processed['TEXT'].apply(stem_words)

# Save processed data to 'data-1.csv'
data_processed.to_csv('data-1.csv', index=False)

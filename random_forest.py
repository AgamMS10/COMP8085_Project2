import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# Load all three datasets
train_data = load_data('Data\train.json')
val_data = load_data('Data\val.json')
test_data = load_data('Data\test.json')



train_texts = [record['text'] for record in train_data]
val_texts = [record['text'] for record in val_data]
test_texts = [record['text'] for record in test_data]


# Get term frequency
# Limit to 5000 max words
vectorizer = TfidfVectorizer(max_features=5000)  


X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)




targets = ['stars', 'useful', 'funny', 'cool']


# Train and Evaluate each target
for target in targets:
    
    # Take the label needed for current target
    train_labels = [record[target] for record in train_data]
    val_labels = [record[target] for record in val_data]
    test_labels = [record[target] for record in test_data]

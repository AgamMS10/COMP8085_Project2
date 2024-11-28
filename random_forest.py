import json
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# Load all three datasets
train_data = load_data('Data\train.json')
val_data = load_data('Data\val.json')
test_data = load_data('Data\test.json')


# Extract text and stars
train_texts = [record['text'] for record in train_data]
train_labels = [record['stars'] for record in train_data]

val_texts = [record['text'] for record in val_data]
val_labels = [record['stars'] for record in val_data]

test_texts = [record['text'] for record in test_data]
test_labels = [record['stars'] for record in test_data]


# Get term frequency
# Limit to 5000 max words
vectorizer = TfidfVectorizer(max_features=5000)  

X_train = vectorizer.fit_transform(train_texts)


X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)




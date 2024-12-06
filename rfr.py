import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

train_data = load_data('/kaggle/input/yelp-dataset-preprocessed/train.json')
val_data = load_data('/kaggle/input/yelp-dataset-preprocessed/val.json')
test_data = load_data('/kaggle/input/yelp-dataset-preprocessed/test.json')


print("Extracting text and transforming with TF-IDF...")
train_texts = [record['text'] for record in train_data]
val_texts = [record['text'] for record in val_data]
test_texts = [record['text'] for record in test_data]


vectorizer = TfidfVectorizer(max_features=4000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

print("\nStarting training for multi-output regression (useful, funny, cool)...")

train_labels = [[record['useful'], record['funny'], record['cool']] for record in train_data]
val_labels = [[record['useful'], record['funny'], record['cool']] for record in val_data]
test_labels = [[record['useful'], record['funny'], record['cool']] for record in test_data]


multi_output_model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=50,       
    max_depth=10,          
    min_samples_split=5,  
    min_samples_leaf=2,    
    random_state=42,
    n_jobs=-1             
))


print("Training multi-output regression model...")
multi_output_model.fit(X_train, train_labels)
print("Training completed for multi-output regression.")


print("Evaluating multi-output regression model...")
val_predictions = multi_output_model.predict(X_val)
test_predictions = multi_output_model.predict(X_test)

for i, target in enumerate(['useful', 'funny', 'cool']):
    val_rmse = mean_squared_error([v[i] for v in val_labels], [v[i] for v in val_predictions], squared=False)
    test_rmse = mean_squared_error([t[i] for t in test_labels], [t[i] for t in test_predictions], squared=False)
    val_r2 = r2_score([v[i] for v in val_labels], [v[i] for v in val_predictions])
    test_r2 = r2_score([t[i] for t in test_labels], [t[i] for t in test_predictions])

    print(f"\nTarget: {target}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Validation R2 Score: {val_r2:.2f}")
    print(f"Test R2 Score: {test_r2:.2f}")


multi_output_model_filename = '/kaggle/working/multi_output_regression_model.pkl'
print(f"Saving multi-output regression model to {multi_output_model_filename}...")
with open(multi_output_model_filename, 'wb') as model_file:
    pickle.dump(multi_output_model, model_file)

print("\nMulti-output regression task completed!")
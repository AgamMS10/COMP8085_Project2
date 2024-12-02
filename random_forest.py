import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# Load datasets
train_data = load_data('/kaggle/input/yelp-dataset-preprocessed/train.json')
val_data = load_data('/kaggle/input/yelp-dataset-preprocessed/val.json')
test_data = load_data('/kaggle/input/yelp-dataset-preprocessed/test.json')

# Extract text and apply TF-IDF
print("Extracting text and transforming with TF-IDF...")
train_texts = [record['text'] for record in train_data]
val_texts = [record['text'] for record in val_data]
test_texts = [record['text'] for record in test_data]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=4000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# Extract labels
train_labels = [record['stars'] for record in train_data]
val_labels = [record['stars'] for record in val_data]
test_labels = [record['stars'] for record in test_data]

print("\nStarting grid search for hyperparameter tuning...")

# Define hyperparameter grid
max_depth_values = [10, 15, 20]
n_estimators_values = [50, 100, 150]
results = []

# Loop through combinations of max_depth and n_estimators
for max_depth in max_depth_values:
    for n_estimators in n_estimators_values:
        print(f"Training with max_depth={max_depth}, n_estimators={n_estimators}...")
        
        # Model params
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, train_labels)
        
        # Validate the model
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save results
        results.append({
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'val_accuracy': val_accuracy
        })


results_df = pd.DataFrame(results)

# Display results
results_df = results_df.sort_values(by='val_accuracy', ascending=False)
print("Grid search results:")
print(results_df)

# Create graph for accuracy with model
plt.figure(figsize=(10, 6))
for max_depth in max_depth_values:
    subset = results_df[results_df['max_depth'] == max_depth]
    plt.plot(subset['n_estimators'], subset['val_accuracy'], label=f'max_depth={max_depth}')

plt.title('Validation Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.savefig('/kaggle/working/validation_accuracy_vs_estimators.png', dpi=300)  # Save the plot
plt.show()

# Find best model
best_params = results_df.iloc[0]
best_model = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, train_labels)
print(f"Best Model Parameters: max_depth={best_params['max_depth']}, n_estimators={best_params['n_estimators']}")

# Test best model
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy for best model: {test_accuracy:.4f}")
print(f"Test Classification Report:\n{classification_report(test_labels, test_predictions)}")

# Confusion Matrix best model
print("Generating confusion matrix for the best model...")
conf_matrix = confusion_matrix(test_labels, test_predictions)

# Save matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Confusion Matrix - Best Model on Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix_best_model.png', dpi=300)  # Save the plot
plt.show()

# Validation accuracy vs test accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(len(results_df)), results_df['val_accuracy'], label='Validation Accuracy', marker='o')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy (Best Model): {test_accuracy:.4f}')
plt.title('Validation vs Test Accuracy')
plt.xlabel('Grid Search Model Index')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('/kaggle/working/validation_vs_test_accuracy.png', dpi=300)  # Save the plot
plt.show()

# Save best model
model_filename = '/kaggle/working/best_random_forest_model.pkl'
print(f"Saving best model to {model_filename}...")
with open(model_filename, 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("Grid search and best model training completed!")

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

train_data = load_data('/kaggle/input/yelp-dataset-preprocessed/train.json')
val_data = load_data('/kaggle/input/yelp-dataset-preprocessed/val.json')
test_data = load_data('/kaggle/input/yelp-dataset-preprocessed/test.json')

train_texts = [record['text'] for record in train_data]
val_texts = [record['text'] for record in val_data]
test_texts = [record['text'] for record in test_data]

vectorizer = TfidfVectorizer(max_features=4000)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

train_labels = [record['stars'] for record in train_data]
val_labels = [record['stars'] for record in val_data]
test_labels = [record['stars'] for record in test_data]

max_depth_values = [10, 15, 20]
n_estimators_values = [50, 100, 150]
results = []

for max_depth in max_depth_values:
    for n_estimators in n_estimators_values:
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
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        results.append({
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'val_accuracy': val_accuracy
        })

results_df = pd.DataFrame(results).sort_values(by='val_accuracy', ascending=False)

plt.figure(figsize=(10, 6))
for max_depth in max_depth_values:
    subset = results_df[results_df['max_depth'] == max_depth]
    plt.plot(subset['n_estimators'], subset['val_accuracy'], label=f'max_depth={max_depth}')
plt.title('Validation Accuracy vs. Number of Estimators (Random Forest)')
plt.xlabel('Number of Estimators')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.savefig('/kaggle/working/validation_accuracy_vs_estimators_rf.png', dpi=300)
plt.show()

best_params = results_df.iloc[0]
best_rf_model = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
best_rf_model.fit(X_train, train_labels)
rf_test_predictions = best_rf_model.predict(X_test)
rf_test_accuracy = accuracy_score(test_labels, rf_test_predictions)

conf_matrix_rf = confusion_matrix(test_labels, rf_test_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix_rf.png', dpi=300)
plt.show()

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='saga'
)
logistic_model.fit(X_train, train_labels)
val_predictions_lr = logistic_model.predict(X_val)
val_accuracy_lr = accuracy_score(val_labels, val_predictions_lr)
lr_test_predictions = logistic_model.predict(X_test)
lr_test_accuracy = accuracy_score(test_labels, lr_test_predictions)

conf_matrix_lr = confusion_matrix(test_labels, lr_test_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Greens', xticklabels=logistic_model.classes_, yticklabels=logistic_model.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix_lr.png', dpi=300)
plt.show()

comparison_df = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Validation Accuracy": [results_df.iloc[0]['val_accuracy'], val_accuracy_lr],
    "Test Accuracy": [rf_test_accuracy, lr_test_accuracy]
})

comparison_df.set_index("Model")[["Validation Accuracy", "Test Accuracy"]].plot(kind='bar', rot=0, figsize=(10, 6))
plt.title('Validation and Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('/kaggle/working/accuracy_comparison.png', dpi=300)
plt.show()

model_filename_rf = '/kaggle/working/best_random_forest_model.pkl'
with open(model_filename_rf, 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

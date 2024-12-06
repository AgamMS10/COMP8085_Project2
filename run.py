import pickle
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

def run_inference(data, labels, model_path, vectorizer_path, model_type=None):
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))

    X = vectorizer.transform(data)
    predictions = model.predict(X)

    if model_type is None:
        from sklearn.base import ClassifierMixin, RegressorMixin
        model_type = "classifier" if isinstance(model, ClassifierMixin) else "regressor"

    if model_type == "classifier":
        print("Classification Report:")
        print(classification_report(labels, predictions))
    elif model_type == "regressor":
        print("Regression Metrics for Multi-Output Targets:")
        if isinstance(labels[0], (list, tuple)):  
            for i, target in enumerate(['useful', 'funny', 'cool']):
                target_mse = mean_squared_error([row[i] for row in labels], [row[i] for row in predictions])
                target_r2 = r2_score([row[i] for row in labels], [row[i] for row in predictions])
                print(f"\nTarget: {target}")
                print(f"Mean Squared Error: {target_mse:.4f}")
                print(f"R2 Score: {target_r2:.4f}")
        else: 
            mse = mean_squared_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")
    return predictions



def main():
    parser = argparse.ArgumentParser(description="Yelp Sentiment Analysis - Inference Only")
    parser.add_argument('--test', type=str, help="Path to the test data (JSONL format).", required=True)
    parser.add_argument('--model', type=str, help="Path to a pre-trained model.", required=True)
    parser.add_argument('--vectorizer', type=str, help="Path to a pre-trained vectorizer.", required=True)
    parser.add_argument('--model_type', type=str, choices=['classifier', 'regressor'], help="Specify whether the model is a classifier or a regressor.")
    args = parser.parse_args()


    test_data = pd.read_json(args.test, lines=True)
    X_test = test_data['text']
    if args.model_type == "classifier":
        y_test = test_data['stars']
    elif args.model_type == "regressor":
        y_test = [[record['useful'], record['funny'], record['cool']] for record in test_data.to_dict(orient='records')]

  
    predictions = run_inference(X_test, y_test, model_path=args.model, vectorizer_path=args.vectorizer, model_type=args.model_type)



if __name__ == "__main__":
    main()

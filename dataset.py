import string
import nltk
import json
import random
from nltk.corpus import stopwords
nltk.download('stopwords')

def process_record(record):

    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans('', '', string.punctuation)

    # Ensure the record has all necessary fields with default values
    processed_record = {
        'stars': record.get('stars', 0),
        'useful': record.get('useful', 0),
        'funny': record.get('funny', 0),
        'cool': record.get('cool', 0),
        'text': record.get('text', ''),
    }

    text = processed_record['text'].lower()
    # Remove punctuation
    text = text.translate(punctuation_table)
    # Remove stop words
    text = " ".join([word for word in text.split() if word not in stop_words])
    processed_record['text'] = text

    return processed_record

def save_data_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for record in data:
            file.write(json.dumps(record) + '\n')

def open_json_file(input_file):
    try:
        records = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                records.append(json.loads(line))
        return records
    except Exception as e:
        print(f"An error occurred: {e}")


def balance_data(data, target):
    class_data = {}

    for record in data:
        classification = record[target]
        if classification not in class_data:
            class_data[classification] = []
        class_data[classification].append(record)

    min_samples = min(len(class_data[target]) for target in class_data)
    balanced_data = []
    for target in class_data:
        balanced_data.extend(class_data[target][:min_samples])

    return balanced_data

def split_json_file(input_file, output_train, output_test, output_val):

    records = open_json_file(input_file)

    balance_records = balance_data(records, 'stars')

    processed_records = [process_record(record) for record in balance_records]

    # Shuffle the data
    random.shuffle(processed_records)

    # Split the data
    total_records = len(processed_records)
    train_end = int(total_records * 0.7)
    test_end = train_end + int(total_records * 0.2)

    train_records = processed_records[:train_end]
    test_records = processed_records[train_end:test_end]
    val_records = processed_records[test_end:]

    # Save to respective files
    print(f"Total records: {total_records}")
    print(f"Train records: {len(train_records)}")
    save_data_to_file(train_records, output_train)

    print(f"Test records: {len(test_records)}")
    save_data_to_file(test_records, output_test)

    print(f"Validation records: {len(val_records)}")
    save_data_to_file(val_records, output_val)

def main():
    split_json_file("E:\8085_project_2\COMP8085_Project2\yelp_academic_dataset_review.json", 'train.json', 'test.json', 'val.json')



if __name__ == '__main__':
    main()  

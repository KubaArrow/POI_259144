import os

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


class Classifier:
    input_dir = "output_csv"

    def __init__(self, features=None, labels=None):
        self.features = features
        self.labels = labels

    def train_test_split(self, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        return clf

    def test(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

    def classify(self, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = self.train_test_split(test_size=test_size, random_state=random_state)
        clf = self.train(X_train, y_train)
        self.test(clf, X_test, y_test)

    def read_features_from_csv(self):
        output_dir = "output_csv"
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        for i, file in enumerate(csv_files):
            print(f"{i + 1}. {file}")
        file_index = int(input("Select CSV file by entering the corresponding number: ")) - 1
        selected_csv = os.path.join(output_dir, csv_files[file_index])
        df = pd.read_csv(selected_csv)
        feature_names = df.columns[:-1]  # Kolumny z cechami (bez ostatniej kolumny z etykietami)
        label_name = df.columns[-1]  # Nazwa kolumny z etykietami
        features = df[feature_names].values
        labels = df[label_name].values
        return features, labels

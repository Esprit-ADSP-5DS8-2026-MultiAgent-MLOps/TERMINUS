import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def train(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = df['target']
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('accuracy', accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == '__main__':
    train('data.csv')

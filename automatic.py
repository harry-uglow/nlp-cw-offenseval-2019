import os
import pandas as pd
import numpy as np
from utils import clean_str
from sklearn import preprocessing, metrics

from autokeras.text.text_supervised import TextClassifier


def preprocess_data(data, tweet_column, label_column):
    y = data.iloc[:, label_column].astype(str)
    lst = np.array(data.values.tolist())
    x = [clean_str(twt) for twt in list(lst[:, tweet_column])]
    x = [' '.join(tweet) for tweet in x]
    return [x, y]

def preprocess_data_test(data, tweet_column):
    lst = np.array(data.values.tolist())
    x = [clean_str(twt) for twt in list(lst[:, tweet_column])]
    x = [' '.join(tweet) for tweet in x]
    return x



def train_model(train, test, test_final_path, path='models/bert/task_1/', train_targets_col=2, test_targets_col=1):
    file = os.path.join(path, 'pytorch_model.bin')

    x_train, y_train = preprocess_data(train, 1, train_targets_col)
    x_test, y_test = preprocess_data(test, 0, test_targets_col)

    le = preprocessing.LabelEncoder()
    le.fit(np.append(y_train, y_test))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    if os.path.isfile(file):
        clf = TextClassifier(verbose=True, path=path)
        clf.initialize_y(y_train)
    else:
        clf = TextClassifier(verbose=True, path=path)
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    matrix = metrics.confusion_matrix(y_test, y_pred)
    # print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")
    print("F1 score is: ", 100 * metrics.f1_score(y_pred, y_test, average='macro'), "%")
    print("Confusion matrix: \n", matrix)

    test_out = pd.read_csv(test_final_path, sep='\t')
    print(test_out.head())
    x_test_out = preprocess_data_test(test_out, 1)
    y_pred_out = clf.predict(x_test_out)
    y_pred_out = le.inverse_transform(y_pred_out)
    print(y_pred_out)

    df = pd.DataFrame({'id': test_out.iloc[:, 0], 'label': y_pred_out})
    df.to_csv(test_final_path.replace('test', 'BERT_result').replace('.tsv', '.csv'), header=False, index=False)

if __name__ == "__main__":
    train = pd.read_csv('data/start-kit/training-v1/offenseval-training-v1.tsv', sep='\t')
    test = pd.read_csv('data/start-kit/trial-data/offenseval-trial.txt', header=None, sep='\t')

    # Task 1
    train_model(train, test, 'data/A/testset-taska.tsv', path='models/bert_base_uncased/task_1/', train_targets_col=2, test_targets_col=1)

    # Task 2
    train = train.dropna(subset=['subtask_b'])
    test = test.dropna(subset=[2])

    train_model(train, test, 'data/B/testset-taskb.tsv', path='models/bert_base_uncased/task_2/', train_targets_col=3, test_targets_col=2)

    # Task 3
    train = train.dropna(subset=['subtask_c'])
    test = test.dropna(subset=[3])

    train_model(train, test, 'data/C/test_set_taskc.tsv', path='models/bert_base_uncased/task_3/', train_targets_col=4, test_targets_col=3)

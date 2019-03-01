import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import itertools
from keras import backend as K


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    words = string.split(' ')
    for idx, word in enumerate(words):
        if word == '@USER' or word == 'URL':
            continue
        elif len(word) > 0 and word[0] == '@':
            words[idx] = '@USER'
            continue

        word = re.sub(r'^https?:\/\/.*', 'URL', word)
        word = re.sub(r"[^A-Za-z0-9()@,!?\'\`]", " ", word)
        word = re.sub(r"\'s", " \'s", word)
        word = re.sub(r"\'ve", " have", word)
        word = re.sub(r"n\'t", " not", word)
        word = re.sub(r"\'re", " are", word)
        word = re.sub(r"\'d", " \'d", word)
        word = re.sub(r"\'ll", " will", word)
        word = re.sub(r",", " , ", word)
        word = re.sub(r"!", " ! ", word)
        word = re.sub(r"\(", " \( ", word)
        word = re.sub(r"\)", " \) ", word)
        word = re.sub(r"\?", " \? ", word)
        word = re.sub(r"\s{2,}", " ", word)
        words[idx] = word.strip().lower()
    return words


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_tweets(tweets, padding_word="<PAD/>", sequence_length=None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if sequence_length is None:
        sequence_length = max(len(x) for x in tweets)
    padded_tweets = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        num_padding = sequence_length - len(tweet)
        padded = tweet + [padding_word] * num_padding
        padded_tweets.append(padded)
    return padded_tweets


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > 1]
    vocabulary_inv += ['$']
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['$']
                   for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def preprocess_data(data, tweet_column, column_label):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)

    ints = le.fit_transform(data.loc[:, column_label].astype(str))
    labels = ohe.fit_transform(ints.reshape(len(ints), 1))

    lst = np.array(data.values.tolist())
    tweets = [clean_str(twt) for twt in list(lst[:, tweet_column])]
    padded = pad_tweets(tweets)
    vocabulary, vocabulary_inv = build_vocab(padded)
    x, y = build_input_data(padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, le, ohe]


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def plot_confusion_matrix(cm, classes=None, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def preprocess_test_data(data, vocab, sequence_length, col):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)

    ints = le.fit_transform(data.iloc[:, col].astype(str))
    labels = ohe.fit_transform(ints.reshape(len(ints), 1))

    lst = np.array(data.values.tolist())
    tweets = [clean_str(twt) for twt in list(lst[:, 0])]
    padded = pad_tweets(tweets, sequence_length=sequence_length)

    x, y = build_input_data(padded, labels, vocab)
    return [x, y]

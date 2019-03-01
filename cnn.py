import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def preprocess_data_test(data, tweet_column, vocabulary, sequence_length):
    lst = np.array(data.values.tolist())
    x = [clean_str(twt) for twt in list(lst[:, tweet_column])]
    padded = pad_tweets(x, sequence_length=sequence_length)
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['$']
               for word in sentence] for sentence in padded])
    return x


def train_and_eval(train, test, task_name, col, test_final_path):
    x, y, vocabulary, vocabulary_inv, le, ohe = preprocess_data(train, 1, "subtask_" + task_name)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)

    sequence_length = x.shape[1] # 56
    vocabulary_size = len(vocabulary_inv) # 18765
    embedding_dim = 256
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5

    epochs = 10
    batch_size = 30

    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=y.shape[1], activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    early_stopping = EarlyStopping(patience=2, verbose=1, monitor='val_f1', mode='max', restore_best_weights=True)
    checkpoint = ModelCheckpoint(task_name + 'weights.{epoch:03d}-{val_f1:.4f}.hdf5',
                                 monitor='val_f1', verbose=1, save_best_only=True,
                                 mode='max')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1])
    print("Traning Model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint, early_stopping], validation_data=(X_test, y_test)) # starts training

    print("Evaluating...")
    test_x, test_y = preprocess_test_data(test, vocabulary, sequence_length, col)

    y_pred = model.predict(test_x)

    y_vec = np.argmax(y_pred, axis=1)
    y_test = np.argmax(test_y, axis=1)
    matrix = confusion_matrix(y_test, y_vec)

    plot_confusion_matrix(matrix)
    print('Accuracy: {}'.format(accuracy_score(y_test, y_vec)))
    print('F1 Score: {}'.format(f1_score(y_test, y_vec, average='macro')))

    test_out = pd.read_csv(test_final_path, sep='\t')
    print(test_out.head())
    x_test_out = preprocess_data_test(test_out, 1, vocabulary, sequence_length)
    y_pred_out = model.predict(x_test_out)
    y_pred_out = np.argmax(y_pred_out, axis=1)
    y_pred_out = le.inverse_transform(y_pred_out)
    print(y_pred_out)

    df = pd.DataFrame({'id': test_out.iloc[:, 0], 'label': y_pred_out})
    df.to_csv(test_final_path.replace('test', 'cnn_result').replace('.tsv', '.csv'), header=False, index=False)


print('Loading data')

# Task A
train_A = pd.read_csv('data/start-kit/training-v1/offenseval-training-v1.tsv', sep='\t')
test_A = pd.read_csv('data/start-kit/trial-data/offenseval-trial.txt', header=None, sep='\t')

# train_and_eval(train_A, test_A, "a", 1, 'data/A/testset-taska.tsv')

# Task B
train_B = train_A.dropna(subset=['subtask_b'])
test_B = test_A.dropna(subset=[2])

# train_and_eval(train_B, test_B, "b", 2, 'data/B/testset-taskb.tsv')

train_C = train_B.dropna(subset=['subtask_c'])
test_C = test_B.dropna(subset=[3])

train_and_eval(train_C, test_C, "c", 3, 'data/C/test_set_taskc.tsv')

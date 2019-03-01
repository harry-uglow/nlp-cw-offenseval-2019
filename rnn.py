import os
import keras
import tensorflow as backend
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Permute, Add
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def mask_focal(alpha=0.25, gamma=2.0):
    def mask_focal_loss(y_true, y_pred):
        # compute the focal loss
        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_factor = backend.where(keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(y_true, y_pred)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(y_true, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return mask_focal_loss

print('Loading data')
train = pd.read_csv('data/start-kit/training-v1/offenseval-training-v1.tsv',
                    sep='\t')
x, y, vocabulary, vocabulary_inv = preprocess_data(train, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)


os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("RNN")

sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
num_filters = 512
drop = 0.5
filter_sizes = [3,4,5]

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
weight_decay = 0.001

inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = ZeroPadding2D(padding=((1, 0), (0, 0)))(conv_1)

conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = ZeroPadding2D(padding=(1, 0))(conv_2)

out = Concatenate()([conv_0, conv_1, conv_2])

out = Reshape((sequence_length - 2, num_filters * 3))(out)

out = Bidirectional(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.25))(out)

outputs = Dense(2, activation="softmax")(out)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_f1:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(patience=2, verbose=1, monitor='val_f1', mode='max')
optimizer = Adam(lr=1e-3, decay=1e-6)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1])

class_weight = {0: 0.24, 1: 0.76}

print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping], validation_data=(X_test, y_test), class_weight=class_weight) # starts training


print("Evaluating...")
test = pd.read_csv('data/start-kit/trial-data/offenseval-trial.txt', header=None, sep='\t')
test_x, test_y = preprocess_test_data(test, vocabulary, sequence_length)

y_pred = model.predict(test_x)

y_vec = np.argmax(y_pred, axis=1)
y_test = np.argmax(test_y, axis=1)
matrix = confusion_matrix(y_test, y_vec)

plot_confusion_matrix(matrix)
print('Accuracy: {}'.format(accuracy_score(y_test, y_vec)))
print('F1 Score: {}'.format(f1_score(y_test, y_vec)))

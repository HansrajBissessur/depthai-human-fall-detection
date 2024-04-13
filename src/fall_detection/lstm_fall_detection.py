import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('../../data/extracted_data')
LOG_DIR = os.path.join('../../Logs')


class LSTMFallDetection:

    def __init__(self, learning_rate=0.001, input_shape=(55, 52)):
        self.tb_callback = TensorBoard(log_dir=LOG_DIR)
        self.model = self._build_model(learning_rate, input_shape)

    def get_model_summary(self):
        print(self.model.summary())

    def fit_model(self, verbose=1, epochs=500):
        x_train, x_test, y_train, y_test = self._initialize_train_test()
        class_weights = {0: 0.3 / 0.7, 1: 0.4 / 0.7}
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=verbose,
                       class_weight=class_weights, callbacks=[self.tb_callback])
        self.model.save('model_weights/action3.h5')

    def load_model_weights(self, weights):
        self.model.load_weights(weights)

    def predict_model(self, sequence):
        return self.model.predict(np.expand_dims(sequence, axis=0))[0]

    def _build_model(self, learning_rate, input_shape):
        model = Sequential()
        model.add(LSTM(48, return_sequences=True, activation='tanh', input_shape=input_shape))
        model.add(LSTM(32, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(LSTM(32, return_sequences=False, activation='tanh'))

        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        adam = Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

    def _get_file_count(self, dir_path):
        count = 0
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)) or os.path.isdir(os.path.join(dir_path, path)):
                count += 1
        return count

    def _initialize_train_test(self, test_size=0.2):
        actions = np.array(['fall', 'not_fall'])
        sequences, labels = [], []
        for idx, action in enumerate(actions):
            no_sequences = self._get_file_count(os.path.join(DATA_PATH, action))

            for sequence in range(1, no_sequences + 1):
                window = []
                sequence_length = self._get_file_count(os.path.join(DATA_PATH, action, str(sequence)))

                for frame_num in range(0, sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(np.array(window))
                labels.append(idx)
        x = np.array(sequences)
        y = np.array(labels)
        print(x.shape)
        return train_test_split(x, y, test_size=test_size)

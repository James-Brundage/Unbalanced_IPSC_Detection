
# Imports
import os
import tqdm
import numpy as np
import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import roc_curve


def experiment_prep (pkl_pth, limit, key=1341, test_size=0.2):
    """
    This function will take a directory name that contains the pickles as formatted in the original experiments, read
    them in, perform a train test split, and standardization.
    :param pkl_pth: Path to the directory that contains the pickles.
    :param limit: Number of pickles that will be read in.
    :param key: Random key for the train test split.
    :param test_size: testing sizze for the train test split.
    :return: Results of the standardized train test split.
    """

    # Read in select files
    pkl_pth_lst = [os.path.join(pkl_pth, p) for p in os.listdir(pkl_pth)]

    # Read in the pickles!
    lst = []
    for p in tqdm.tqdm(pkl_pth_lst[:limit]):
        arr = np.load(p, allow_pickle=True)
        lst.append(arr)

    # Create one array
    datarr = np.asarray(lst).reshape(len(lst), 4004)

    # Grab X and y
    X = datarr[:, 1:4001]
    y = datarr[:, -1]
    y=y.astype(float)

    # TTS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=key)

    # Standardize X
    mm_scaler = preprocessing.StandardScaler()
    X_train = X_train_minmax = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def dnn_experiments (X_train, X_test, y_train, y_test, model_key, metrics=['accuracy', 'Precision', 'Recall', 'AUC'], save=False, name='Model'):

    # Initialize input values
    n_class = 2
    n_features = X_train[0].shape[0]
    n_sample = X_train.shape[0]
    array_dim = (n_features,)

    # Set Architectures
    base_model = Sequential()
    base_model.add(Dense(2000, input_dim=n_features, activation='relu'))
    base_model.add(Dense(200, activation='relu'))
    base_model.add(Dense(200, activation='relu'))
    base_model.add(Dense(20, activation='relu'))
    base_model.add(Dense(20, activation='relu'))
    base_model.add(Dense(20, activation='relu'))
    base_model.add(Dense(1, activation='sigmoid'))

    cnn_model = Sequential()
    cnn_model.add(Conv1D(200, kernel_size=1, activation='relu', input_shape=(n_features, 1)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(200, activation='relu'))
    cnn_model.add(Dense(200, activation='relu'))
    cnn_model.add(Dense(20, activation='relu'))
    cnn_model.add(Dense(20, activation='relu'))
    cnn_model.add(Dense(20, activation='relu'))
    cnn_model.add(Dense(1, activation='sigmoid'))

    # Set model dct
    model_dct = {'FCNN': base_model,
                 'CNN': cnn_model}

    # Select model
    model = model_dct[model_key]

    # CNN reshape
    if model_key == 'CNN':
        print(X_train.shape)
        X_train = X_train.reshape((n_sample,
                n_features,
                1))

        n_features_test = X_test[0].shape[0]
        n_sample_test = X_test.shape[0]

        X_test = X_test.reshape((n_sample_test,
                n_features_test,
                1))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    # Fit model
    start = time.time()
    history = model.fit(X_train, y_train, epochs=3, validation_split=0.1)
    end = time.time()
    tt = end-start

    # Grab relevant scores from evaluataion on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)

    # Grab the ROC curve data
    preds = model.predict(X_test)
    fpr, tpr, thresh = roc_curve(y_test, preds, pos_label=1)

    # Saving
    if save == True:
        filename = str(name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))

    return scores, tt, history.history, fpr, tpr, thresh


# Test Code
# pkl_rt = '/Users/jamesbrundage/PycharmProjects/IPSC_Detector/Datasets/CNN_pkls'
# X_train, X_test, y_train, y_test = experiment_prep(pkl_rt, 100)
# scores, tt, history, fpr, tpr, thresh = dnn_experiments(X_train, X_test, y_train, y_test, 'FCNN')

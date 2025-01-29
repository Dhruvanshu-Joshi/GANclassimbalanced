import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score
from imblearn.metrics import geometric_mean_score
from statistics import stdev

class test_model():  # Parent class
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Using sigmoid for binary classification
        ])
    
        model.compile(optimizer='adam',  # Configures the model for training
                      loss='binary_crossentropy',  # Changed loss to binary_crossentropy
                      metrics=['accuracy'])
        
        model.fit(self.X_train, self.y_train, epochs=30, verbose=0)

        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        train_loss, train_accuracy = model.evaluate(self.X_train, self.y_train, verbose=0)
        
        y_preds = model.predict(self.X_test)
        y_preds = np.ravel((y_preds > 0.5) * 1)
        
        F1_score = f1_score(self.y_test, y_preds, average='micro')
        AP_score = average_precision_score(self.y_test, y_preds)
        G_mean = geometric_mean_score(self.y_test, y_preds)

        return test_accuracy, train_accuracy, F1_score, AP_score, G_mean


def test_model_lists(X_train, y_train, X_test, y_test, no_of_trainings):
    test_accuracy_array = []
    train_accuracy_array = []
    f1_score_array = []
    ap_score_array = []
    gmean_array = []
    
    test_model_object = test_model(X_train, y_train.ravel(), X_test, y_test.ravel())

    for i in range(no_of_trainings):
        test_accuracy, train_accuracy, F1_score, AP_score, G_mean = test_model_object()
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)
        f1_score_array.append(F1_score)
        ap_score_array.append(AP_score)
        gmean_array.append(G_mean)
    
    return test_accuracy_array, train_accuracy_array, f1_score_array, ap_score_array, gmean_array
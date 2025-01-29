import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from statistics import stdev

from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from imblearn.metrics import geometric_mean_score

class test_model():        
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for probability output
        ])
    
        model.compile(optimizer='adam',               
                      loss='binary_crossentropy',  # Use binary cross-entropy for classification
                      metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=30, verbose=0)

        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        train_loss, train_accuracy = model.evaluate(self.X_train, self.y_train, verbose=0)
        
        y_probs = model.predict(self.X_test).ravel()  # Get probability scores
        y_preds = (y_probs > 0.5).astype(int)  # Convert probabilities to binary predictions
        
        F1_score = f1_score(self.y_test, y_preds, average='micro')  
        AP_score = average_precision_score(self.y_test, y_probs.reshape(-1, 1))  # Reshape to 2D
        G_mean = geometric_mean_score(self.y_test, y_preds)  # G-Mean
        AUC_score = roc_auc_score(self.y_test, y_probs)  # **AUC Score**

        return test_accuracy, train_accuracy, F1_score, AP_score, G_mean, AUC_score





def test_model_lists(X_train, y_train, X_test, y_test, no_of_trainings):
    test_accuracy_array = []
    train_accuracy_array = []
    f1_score_array = []
    ap_score_array = []
    gmean_array = []
    auc_array=[]
    test_model_object = test_model(X_train, y_train.ravel(), X_test, y_test.ravel())

    for i in range(no_of_trainings):
        test_accuracy, train_accuracy, F1_score, AP_score, G_mean, AUC_score = test_model_object()
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)
        f1_score_array.append(F1_score)
        ap_score_array.append(AP_score)
        gmean_array.append(G_mean)
        auc_array.append(AUC_score)
    return test_accuracy_array, train_accuracy_array, f1_score_array, ap_score_array, gmean_array, auc_array
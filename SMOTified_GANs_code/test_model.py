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
        self.num_classes = len(np.unique(y_train))  # Count unique classes

    def __call__(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1 if self.num_classes == 2 else self.num_classes, activation='sigmoid' if self.num_classes == 2 else 'softmax')
        ])
    
        model.compile(optimizer='adam',               
                      loss='binary_crossentropy' if self.num_classes == 2 else 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=30, verbose=0)

        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        train_loss, train_accuracy = model.evaluate(self.X_train, self.y_train, verbose=0)
        
        y_probs = model.predict(self.X_test)  # Get probability scores
        
        # Convert probabilities to predictions
        if self.num_classes == 2:
            y_preds = (y_probs > 0.5).astype(int).flatten()
            y_probs = y_probs.flatten()  # Ensure y_probs is 1D for binary case
        else:
            y_preds = np.argmax(y_probs, axis=1)  # Take argmax for multi-class

        F1_score = f1_score(self.y_test, y_preds, average='micro')  
        AP_score = average_precision_score(self.y_test, y_probs if self.num_classes == 2 else y_probs, average='micro')  
        G_mean = geometric_mean_score(self.y_test, y_preds)  

        # Handle AUC correctly
        if self.num_classes == 2:
            AUC_score = roc_auc_score(self.y_test, y_probs)  # Binary case (1D probs)
        else:
            AUC_score = roc_auc_score(self.y_test, y_probs, multi_class='ovr')  # Multi-class (2D probs)

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
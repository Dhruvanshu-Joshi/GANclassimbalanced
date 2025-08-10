from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import torch
import os
import time
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from dataset_loader import load_dataset
from GANs_model import GANs_Discriminator, GANs_Generator
from model_trainer import train_discriminator, train_generator
from model_fit import SG_fit, G_fit
from test_model import test_model, test_model_lists
from choose_device import get_default_device, to_device
from fit import f1
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.datasets import load_wine
# from real_data_generator import get_real_data_for_GAN  # Import the function
import os
import time

# Keep your imports from original code

DATASETS = dict()

"""Wine Dataset"""
X, y = load_wine(return_X_y=True)
DATASETS.update({
    'Wine': {
        'data': [X, y],
        'extra': {
        }
    }
})

"""Flare-F"""
data = pd.read_csv('data/raw/flare-F.dat', header=None)
objects = data.select_dtypes(include=['object'])
for col in objects.columns:
    if col == len(data.columns) - 1:
        continue
    data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])

DATASETS.update({
    'Flare-F': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {

        }
    }
})

"""Yeast5"""
data = pd.read_csv('data/raw/yeast5.dat', header=None)
DATASETS.update({
    'Yeast5': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Car vGood"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarvGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'vgood'
        }
    }
})


"""Car Good"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'good'
        }
    }
})

"""Seed"""
data = pd.read_csv('data/raw/seeds_dataset.txt', header=None)
DATASETS.update({
    'Seed': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 2
        }
    }
})

"""Glass"""
data = pd.read_csv('data/raw/glass.csv', header=None)
DATASETS.update({
    'Glass': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 7
        }
    }
})

"""ILPD"""
data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
# Loop through columns
for col in data.columns:
    # Get unique non-null values
    unique_vals = set(data[col].dropna().unique())
    
    # Check if the column is purely Male/Female
    if unique_vals <= {"Male", "Female"}:
        data[col] = data[col].map({"Male": 1, "Female": 0})
data.fillna(data.mean(), inplace=True)

#Encode
data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

DATASETS.update({
    'ILPD': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Yeast5-ERL"""
data = pd.read_csv('data/raw/yeast5.data', header=None)
DATASETS.update({
    'Yeast5-ERL': {
        'data': [data.values[:, 1:-1], data.values[:, -1]],
        'extra': {
            # 'minority_class': 'ME1'
            'minority_class': 'ERL'
        }
    }
})

# Load the Epileptic Seizure Recognition dataset
data = pd.read_csv('data/raw/seizure.csv', header=0, low_memory=False)

DATASETS.update({
    'Epileptic Seizure Recognition': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})


# Load the breast cancer dataset
data = pd.read_csv('data/raw/breast_cancer.csv', header=None)

# Encode categorical features if necessary
objects = data.select_dtypes(include=['object'])
for col in objects.columns:
    if col == data.shape[1] - 1:  # Skip the last column if it's the target
        continue
    data.iloc[:, col] = LabelEncoder().fit_transform(data.iloc[:, col])

# Update the DATASETS dictionary
DATASETS.update({
    'Breast Cancer Wisconsin': {
        'data': [data.iloc[:, :-1].values, data.iloc[:, -1].values],  # Features and target
        'extra': {

        }
    }
})


'''Diabetes'''
data = pd.read_csv('data/raw/diabetes_data.csv', header=0)
# Loop through columns
# Step 1: Convert 'Male'/'Female' to 1/0 where applicable
for col in data.columns:
    unique_vals = set(data[col].dropna().unique())
    if unique_vals <= {"Male", "Female"}:
        data[col] = data[col].map({"Male": 1, "Female": 0})

# Step 2: One-hot encode all remaining object/string columns
categorical_cols = data.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Step 3: Fill missing numeric values with column means
data.fillna(data.mean(numeric_only=True), inplace=True)

DATASETS.update({
    'Diabetes': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})


'''sonar'''
data = pd.read_csv('data/raw/sonar_all_data.csv', header=None)

DATASETS.update({
    'Sonar': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})


'''student_dropout'''
data = pd.read_csv('data/raw/student_dropout.csv', header=0)

DATASETS.update({
    'Student_dropout': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})


'''default of credit card clients'''
data = pd.read_excel('data/raw/default of credit card clients.xls', header=0)

DATASETS.update({
    'default of credit card clients': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})

# # Function to shuffle data
# def shuffle_in_unison(a, b):
#     assert len(a) == len(b)
#     permutation = np.random.permutation(len(a))
#     return a[permutation], b[permutation]

# def main():
#     device = get_default_device()
#     output_dir = "results"
#     os.makedirs(output_dir, exist_ok=True)

#     n_runs = 20
#     k_folds = 5

#     for dataset_name, dataset in DATASETS.items():
#         print(f"Processing {dataset_name}...")
#         X, y = dataset['data']
#         y = y - 1  # Adjust labels if required
#         X = X.astype(float)
#         y = y.astype(float)

#         excel_rows = []  # to store per-run per-fold metrics

#         for run in range(n_runs):
#             kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=run)

#             for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
#                 X_train, X_test = X[train_idx], X[test_idx]
#                 y_train, y_test = y[train_idx], y[test_idx]

#                 # SMOTE oversampling
#                 X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
#                 X_oversampled = torch.tensor(X_train_SMOTE[len(X_train):], dtype=torch.float32).to(device)

#                 lr, epochs, batch_size = 0.0002, 150, 128

#                 # ---- SG_GANs ----
#                 start_time_sg = time.time()
#                 generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE,
#                                                X_train, y_train, X_oversampled,
#                                                device, lr, epochs, batch_size, 1, 0)

#                 X_trained_SG = generator_SG(X_oversampled).cpu().detach().numpy()
#                 X_final_SG, y_final_SG = shuffle_in_unison(
#                     np.vstack((X_train_SMOTE[:len(X_train)], X_trained_SG)),
#                     y_train_SMOTE
#                 )

#                 test_acc, train_acc, f1_score, ap_score, gmean_score, auc_score, \
#                 std_dev_acc, std_dev_f1, std_dev_ap, std_dev_gmean, std_dev_auc = \
#                     test_model_lists(X_final_SG, y_final_SG, X_test, y_test, 1)  # 1 run here

#                 sg_time_taken = time.time() - start_time_sg

#                 excel_rows.append({
#                     "Dataset": dataset_name,
#                     "Run": run + 1,
#                     "Fold": fold + 1,
#                     "Method": "SG_GANs",
#                     "Accuracy": np.mean(test_acc),
#                     "AUC": np.mean(auc_score),
#                     "F1": np.mean(f1_score),
#                     "AP": np.mean(ap_score),
#                     "GMean": np.mean(gmean_score),
#                     "TimeTaken": sg_time_taken
#                 })

#                 # ---- G_GANs ----
#                 start_time_g = time.time()
#                 generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE,
#                                                X_train, y_train, X_oversampled,
#                                                device, lr, epochs, batch_size, 1, 0)

#                 X_trained_G = generator_G(torch.randn_like(X_oversampled)).cpu().detach().numpy()
#                 X_final_G, y_final_G = shuffle_in_unison(
#                     np.vstack((X_train_SMOTE[:len(X_train)], X_trained_G)),
#                     y_train_SMOTE
#                 )

#                 test_acc, train_acc, f1_score, ap_score, gmean_score, auc_score, \
#                 std_dev_acc, std_dev_f1, std_dev_ap, std_dev_gmean, std_dev_auc = \
#                     test_model_lists(X_final_G, y_final_G, X_test, y_test, 1)

#                 g_time_taken = time.time() - start_time_g

#                 excel_rows.append({
#                     "Dataset": dataset_name,
#                     "Run": run + 1,
#                     "Fold": fold + 1,
#                     "Method": "G_GANs",
#                     "Accuracy": np.mean(test_acc),
#                     "AUC": np.mean(auc_score),
#                     "F1": np.mean(f1_score),
#                     "AP": np.mean(ap_score),
#                     "GMean": np.mean(gmean_score),
#                     "TimeTaken": g_time_taken
#                 })

#         # Save per-run results for this dataset
#         df_results = pd.DataFrame(excel_rows)
#         df_results.to_excel(os.path.join(output_dir, f"results_{dataset_name}.xlsx"), index=False)
#         print(f"Saved detailed results for {dataset_name} to Excel.")

# if __name__ == "__main__":
#     main()

# ---------------- Utility Functions ---------------- #
def shuffle_in_unison(a, b):
    """Shuffle two arrays in the same order."""
    assert len(a) == len(b)
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]

def log_nans(label, arr):
    """Log if NaNs or Infs are detected."""
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"[WARNING] {label} contains NaN/Inf values!")
        print(f"  NaN count: {np.isnan(arr).sum()} | Inf count: {np.isinf(arr).sum()}")

def clean_data(X, y, label_X="", label_y=""):
    """Replace NaNs and Infs in features and labels."""
    log_nans(label_X, X)
    log_nans(label_y, y)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(float), y.astype(float)

# ---------------- Main Training Loop ---------------- #
def main():
    device = get_default_device()
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    n_runs = 20
    k_folds = 5

    for dataset_name, dataset in DATASETS.items():
        print(f"Processing {dataset_name}...")
        X, y = dataset['data']
        y = y - 1  # Adjust labels if required
        X, y = clean_data(X, y, "Initial X", "Initial y")

        excel_rows = []  # Store per-run per-fold metrics

        for run in range(n_runs):
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=run)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train, y_train = clean_data(X_train, y_train, "X_train", "y_train")
                X_test, y_test = clean_data(X_test, y_test, "X_test", "y_test")

                # SMOTE oversampling
                X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
                X_train_SMOTE, y_train_SMOTE = clean_data(
                    X_train_SMOTE, y_train_SMOTE, "X_train_SMOTE", "y_train_SMOTE"
                )

                X_oversampled = torch.tensor(
                    X_train_SMOTE[len(X_train):], dtype=torch.float32
                ).to(device)

                lr, epochs, batch_size = 0.0002, 150, 128

                # ---- SG_GANs ----
                start_time_sg = time.time()
                generator_SG, generator_G = f1(
                    X_train, y_train,
                    X_train_SMOTE, y_train_SMOTE,
                    X_train, y_train,
                    X_oversampled,
                    device, lr, epochs, batch_size, 1, 0
                )

                X_trained_SG = generator_SG(X_oversampled).cpu().detach().numpy()
                X_trained_SG, _ = clean_data(X_trained_SG, y_train_SMOTE[len(X_train):], "X_trained_SG", "y_extra_SG")

                X_final_SG, y_final_SG = shuffle_in_unison(
                    np.vstack((X_train_SMOTE[:len(X_train)], X_trained_SG)),
                    y_train_SMOTE
                )
                X_final_SG, y_final_SG = clean_data(X_final_SG, y_final_SG, "X_final_SG", "y_final_SG")

                test_acc, train_acc, f1_s, ap_s, gmean_s, auc_s, \
                std_dev_acc, std_dev_f1, std_dev_ap, std_dev_gmean, std_dev_auc = \
                    test_model_lists(X_final_SG, y_final_SG, X_test, y_test, 1)

                sg_time_taken = time.time() - start_time_sg

                excel_rows.append({
                    "Dataset": dataset_name,
                    "Run": run + 1,
                    "Fold": fold + 1,
                    "Method": "SG_GANs",
                    "Accuracy": np.mean(test_acc),
                    "AUC": np.mean(auc_s),
                    "F1": np.mean(f1_s),
                    "AP": np.mean(ap_s),
                    "GMean": np.mean(gmean_s),
                    "TimeTaken": sg_time_taken
                })

                # ---- G_GANs ----
                start_time_g = time.time()
                generator_SG, generator_G = f1(
                    X_train, y_train,
                    X_train_SMOTE, y_train_SMOTE,
                    X_train, y_train,
                    X_oversampled,
                    device, lr, epochs, batch_size, 1, 0
                )

                X_trained_G = generator_G(torch.randn_like(X_oversampled)).cpu().detach().numpy()
                X_trained_G, _ = clean_data(X_trained_G, y_train_SMOTE[len(X_train):], "X_trained_G", "y_extra_G")

                X_final_G, y_final_G = shuffle_in_unison(
                    np.vstack((X_train_SMOTE[:len(X_train)], X_trained_G)),
                    y_train_SMOTE
                )
                X_final_G, y_final_G = clean_data(X_final_G, y_final_G, "X_final_G", "y_final_G")

                test_acc, train_acc, f1_s, ap_s, gmean_s, auc_s, \
                std_dev_acc, std_dev_f1, std_dev_ap, std_dev_gmean, std_dev_auc = \
                    test_model_lists(X_final_G, y_final_G, X_test, y_test, 1)

                g_time_taken = time.time() - start_time_g

                excel_rows.append({
                    "Dataset": dataset_name,
                    "Run": run + 1,
                    "Fold": fold + 1,
                    "Method": "G_GANs",
                    "Accuracy": np.mean(test_acc),
                    "AUC": np.mean(auc_s),
                    "F1": np.mean(f1_s),
                    "AP": np.mean(ap_s),
                    "GMean": np.mean(gmean_s),
                    "TimeTaken": g_time_taken
                })

        # Save per-run results for this dataset
        df_results = pd.DataFrame(excel_rows)
        df_results.to_excel(os.path.join(output_dir, f"results_{dataset_name}.xlsx"), index=False)
        print(f"Saved detailed results for {dataset_name} to Excel.")

if __name__ == "__main__":
    main()

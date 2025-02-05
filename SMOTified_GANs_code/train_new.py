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

# # Define datasets
DATASETS = dict()

# # Load Wine Dataset
# X, y = load_wine(return_X_y=True)
# DATASETS['Wine'] = {'data': [X, y], 'extra': {}}

# # Load other datasets
# DATASET_PATHS = {
#     # 'Flare-F': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/flare-F.dat',
#     # 'Yeast5': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/yeast5.dat',
#     # 'CarvGood': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/car.data',
#     # 'CarGood': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/car.data',
#     'Seed': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/seeds_dataset.txt',
#     'Glass': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/glass.csv',
#     # 'ILPD': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/Indian Liver Patient Dataset (ILPD).csv',
#     # 'Yeast5-ERL': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/yeast5.data',
#     # 'HIGGS': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/higgs.csv',
#     # 'kdd_cup_new': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/kdd_cup_new.csv',
#     # 'Epileptic Seizure Recognition': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/seizure.csv',
#     'Breast Cancer Wisconsin': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/breast_cancer.csv',
#     # 'Diabetes': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/diabetes_data.csv',
#     'Sonar': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/sonar_all_data.csv',
#     'Student Dropout': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/student_dropout.csv',
#     # 'Credit Card Default': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/default of credit card clients.xls'
# }

# # Yeast
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/yeast5.dat', header=None)
# data.iloc[:, -1] = data.iloc[:, -1].map({'negative': 0, 'positive': 1})
# data.iloc[:, -1] = data.iloc[:, -1].astype(int)
# DATASETS.update({
#     'Yeast5': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })

# """Flare-F"""
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/flare-F.dat', header=None)
# objects = data.select_dtypes(include=['object'])
# for col in objects.columns:
#     if col == len(data.columns) - 1:
#         continue
#     data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.values[:, -1])
# DATASETS.update({
#     'Flare-F': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {

#         }
#     }
# })


# # """Car vGood"""
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/car.data', header=None)
# DATASETS.update({
#     'CarvGood': {
#         'data': [
#             OrdinalEncoder().fit_transform(data.values[:, :-1]),
#             LabelEncoder().fit_transform(data.values[:, -1])
#         ],
#         'extra': {
#             'minority_class': 'vgood'
#         }
#     }
# })

# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/seeds_dataset.txt', header=None)
# DATASETS.update({
#     'Seed': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             'minority_class': 2
#         }
#     }
# })

# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/glass.csv', header=None)

# DATASETS.update({
#     'Glass': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             'minority_class': 7
#         }
#     }
# })


# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)


# #Encode
# data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])
# data.fillna(data.mean(), inplace=True)
# DATASETS.update({
#     'ILPD': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })


# # Load the Epileptic Seizure Recognition dataset
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/seizure.csv', header=0, low_memory=False)

# DATASETS.update({
#     'Epileptic Seizure Recognition': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })



# # Load the breast cancer dataset
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/breast_cancer.csv', header=None)

# # Encode categorical features if necessary
# # objects = data.select_dtypes(include=['object'])
# # for col in objects.columns:
# #     if col == data.shape[1] - 1:  # Skip the last column if it's the target
# #         continue
# #     data.iloc[:, col] = LabelEncoder().fit_transform(data.iloc[:, col])
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])
# # Update the DATASETS dictionary
# DATASETS.update({
#     'Breast Cancer Wisconsin': {
#         'data': [data.values[:, :-1], data.values[:, -1]],  # Features and target
#         'extra': {

#         }
#     }
# })



# '''Diabetes'''
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/diabetes_data.csv', header=0)
# data.iloc[:,:-1]=OrdinalEncoder().fit_transform(data.iloc[:, :-1])
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])
# data.fillna(data.mean(), inplace=True)

# DATASETS.update({
#     'Diabetes': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })



# '''sonar'''
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/sonar_all_data.csv', header=None)

# DATASETS.update({
#     'Sonar': {
#         'data': [data.values[:, :-1], LabelEncoder().fit_transform(data.values[:, -1])],
#         'extra': {}
#     }
# })


# '''student_dropout'''
# data = pd.read_csv('/content/GANclassimbalanced/SMOTified_GANs_code/raw/student_dropout.csv', header=0)

# DATASETS.update({
#     'studen': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })


# Function to shuffle data
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]

# Main function
def main():
    device = get_default_device()
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, dataset in DATASETS.items():
        print(f"Processing {dataset_name}...")
        X, y = dataset['data']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train - 1
        y_test = y_test - 1
        # y_train = np.where(y_train >= 4, y_train - 1, y_train)  # Only modify values 4,5,6
        # y_test = np.where(y_test >= 4, y_test - 1, y_test)      # Keep 0,1,2 unchanged
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        # print("Before OverSampling:", np.bincount(y_train))
        X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
        # print("After OverSampling:", np.bincount(y_train_SMOTE))
        
        X_oversampled = torch.tensor(X_train_SMOTE[len(X_train):], dtype=torch.float32).to(device)
        lr, epochs, batch_size = 0.0002, 150, 128
        
        start_time_sg = time.time()
        generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_train, y_train, X_oversampled, device, lr, epochs, batch_size, 1, 0) 
        X_trained_SG = generator_SG(X_oversampled).cpu().detach().numpy()
        X_final_SG, y_final_SG = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_SG)), y_train_SMOTE)
        metrics = {   
            "SG_GANs": test_model_lists(X_final_SG, y_final_SG, X_test, y_test, 20)
        }
        sg_time_taken = time.time() - start_time_sg

        start_time_g = time.time()
        generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_train, y_train, X_oversampled, device, lr, epochs, batch_size, 1, 0) 
        X_trained_G = generator_G(torch.randn_like(X_oversampled)).cpu().detach().numpy()   
        X_final_G, y_final_G = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_G)), y_train_SMOTE)
        metrics["G_GANs"]= test_model_lists(X_final_G, y_final_G, X_test, y_test, 20) 
        g_time_taken = time.time() - start_time_g
        
        output_path = os.path.join(output_dir, f"{dataset_name}_results.txt")
        with open(output_path, "w") as f:
            for model, (test_acc, train_acc, f1_score,ap_score,gmean_score,auc_score,std_dev_acc, std_dev_f1,std_dev_ap,std_dev_gmean,std_dev_auc) in metrics.items():
                result = f"{model} - Test Acc: {np.mean(test_acc):.4f} ± {std_dev_acc:.4f} , AUC: {np.mean(auc_score):.4f} ± {std_dev_auc:.4f}, F1: {np.mean(f1_score):.4f} ± {std_dev_f1:.4f}, AP: {np.mean(ap_score):.4f} ± {std_dev_ap:.4f}, GMEAN: {np.mean(gmean_score):.4f} ± {std_dev_gmean:.4f}\n"
                print(result)
                print(f"SG_GANs Time Taken: {sg_time_taken:.2f} seconds")
                print(f"G_GANs Time Taken: {g_time_taken:.2f} seconds")
                f.write(result)

if __name__ == "__main__":
    main()
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

# Define datasets
DATASETS = dict()

# Load Wine Dataset
X, y = load_wine(return_X_y=True)
DATASETS['Wine'] = {'data': [X, y], 'extra': {}}

# Load other datasets
DATASET_PATHS = {
    # 'Flare-F': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/flare-F.dat',
    'Yeast5': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/yeast5.dat',
    # 'CarvGood': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/car.data',
    # 'CarGood': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/car.data',
    'Seed': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/seeds_dataset.txt',
    'Glass': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/glass.csv',
    # 'ILPD': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/Indian Liver Patient Dataset (ILPD).csv',
    # 'Yeast5-ERL': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/yeast5.data',
    # 'HIGGS': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/higgs.csv',
    # 'kdd_cup_new': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/kdd_cup_new.csv',
    # 'Epileptic Seizure Recognition': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/seizure.csv',
    'Breast Cancer Wisconsin': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/breast_cancer.csv',
    # 'Diabetes': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/diabetes_data.csv',
    'Sonar': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/sonar_all_data.csv',
    'Student Dropout': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/student_dropout.csv',
    # 'Credit Card Default': '/content/GANclassimbalanced/SMOTified_GANs_code/raw/default of credit card clients.xls'
}

for name, path in DATASET_PATHS.items():
    try:
        if path.endswith('.csv'):
            data = pd.read_csv(path, header=None)
        elif path.endswith('.xls'):
            data = pd.read_excel(path, header=0)
        else:
            data = pd.read_csv(path, delimiter='\t', header=None)

        # Encode categorical features
        objects = data.select_dtypes(include=['object'])
        for col in objects.columns:
            if col != data.shape[1] - 1:
                data.iloc[:, col] = LabelEncoder().fit_transform(data.iloc[:, col])

        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        DATASETS[name] = {'data': [X, y], 'extra': {}}
    except Exception as e:
        print(f"Error loading {name}: {e}")

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
        
        print("Before OverSampling:", np.bincount(y_train.astype(int)))
        X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
        print("After OverSampling:", np.bincount(y_train_SMOTE))
        
        X_oversampled = torch.tensor(X_train_SMOTE[len(X_train):], dtype=torch.float32).to(device)
        
        lr, epochs, batch_size = 0.0002, 150, 128
        generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_train, y_train, X_oversampled, device, lr, epochs, batch_size, 1, 0)
        
        X_trained_SG = generator_SG(X_oversampled).cpu().detach().numpy()
        X_trained_G = generator_G(torch.randn_like(X_oversampled)).cpu().detach().numpy()
        
        X_final_SG, y_final_SG = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_SG)), y_train_SMOTE)
        X_final_G, y_final_G = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_G)), y_train_SMOTE)
        
        metrics = {
            # "Normal": test_model_lists(X_train, y_train, X_test, y_test, 20),
            # "SMOTE": test_model_lists(X_train_SMOTE, y_train_SMOTE, X_test, y_test, 20),
            "SG_GANs": test_model_lists(X_final_SG, y_final_SG, X_test, y_test, 20),
            "G_GANs": test_model_lists(X_final_G, y_final_G, X_test, y_test, 20)
        }
        
        output_path = os.path.join(output_dir, f"{dataset_name}_results.txt")
        with open(output_path, "w") as f:
            for model, (test_acc, train_acc, f1_score,ap_score,gmean_score) in metrics.items():
                result = f"{model} - Test Acc: {np.mean(test_acc)}, F1: {np.mean(f1_score)}, AP: {np.mean(ap_score)}, GMEAN: {np.mean(gmean_score)}\n"
                print(result)
                f.write(result)

if __name__ == "__main__":
    main()
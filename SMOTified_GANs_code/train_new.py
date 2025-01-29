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

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    permutation = np.random.permutation(len(a))
    return a[permutation], b[permutation]

def main():
    dataset_path = input("Enter the dataset path (.csv, .txt, .dat): ")
    dataset_path = "SMOTified_GANs_code/raw/breast_cancer.csv"
    df = load_dataset(dataset_path)
    
    X = df.iloc[:, :-1].values  # Features (all columns except last)
    y = df.iloc[:, -1].values   # Target (last column)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Before OverSampling:", np.bincount(y_train))
    X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
    print("After OverSampling:", np.bincount(y_train_SMOTE))
    
    device = get_default_device()
    X_oversampled = torch.tensor(X_train_SMOTE[len(X_train):], dtype=torch.float32).to(device)
    
    lr, epochs, batch_size = 0.0002, 150, 128
    generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_train, y_train, X_oversampled, device, lr, epochs, batch_size, 1, 0)
    
    X_trained_SG = generator_SG(X_oversampled).cpu().detach().numpy()
    X_trained_G = generator_G(torch.randn_like(X_oversampled)).cpu().detach().numpy()
    
    X_final_SG, y_final_SG = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_SG)), y_train_SMOTE)
    X_final_G, y_final_G = shuffle_in_unison(np.vstack((X_train_SMOTE[:len(X_train)], X_trained_G)), y_train_SMOTE)
    
    metrics = {
        "Normal": test_model_lists(X_train, y_train, X_test, y_test, 30),
        "SMOTE": test_model_lists(X_train_SMOTE, y_train_SMOTE, X_test, y_test, 30),
        "SG_GANs": test_model_lists(X_final_SG, y_final_SG, X_test, y_test, 30),
        "G_GANs": test_model_lists(X_final_G, y_final_G, X_test, y_test, 30)
    }
    
    for model, (test_acc, train_acc, f1_score) in metrics.items():
        print(f"{model} - Test Acc: {test_acc}, Train Acc: {train_acc}, F1: {f1_score}")

if __name__ == "__main__":
    main()

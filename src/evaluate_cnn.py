import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["filename"]
        label = self.df.iloc[idx]["label"]
        img_path = os.path.join(self.image_dir, filename)
        
        # Safe fallback for missing images to prevent crashing
        if not os.path.exists(img_path):
            return torch.randn(1, 224, 224), label
            
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return torch.randn(1, 224, 224), label
            
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class OCTConvNet(nn.Module):
    def __init__(self, out_c1, out_c2, dense_units, n_classes=4, img_size=224):
        super(OCTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_c1, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=out_c1, out_channels=out_c2, kernel_size=5)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = x.numel()

        self.fc1 = nn.Linear(flattened_size, dense_units)
        self.logits = nn.Linear(dense_units, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.logits(x)

def compute_classification_metrics(y_true, y_pred, unique_classes):
    precisions, recalls, f1_scores = [], [], []
    for c in unique_classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    accuracy = np.mean(y_true == y_pred)
    return {
        'accuracy': accuracy,
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores)
    }

def predict_cnn(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    return all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    csv_path = os.path.join(base_dir, "data", "image_metadata.csv")
    models_dir = os.path.join(base_dir, "models")
    
    # Load metadata
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    metadata_df = pd.read_csv(csv_path)
    
    # Filter dataset for only images that actually exist to avoid noise injection overhead
    valid_rows = []
    for _, row in metadata_df.iterrows():
        if os.path.exists(os.path.join(data_dir, row['filename'])):
            valid_rows.append(row)
    metadata_df = pd.DataFrame(valid_rows)
    
    print(f"Total base (clean) images found: {len(metadata_df)}")
    
    if len(metadata_df) == 0:
        return
        
    oct_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Prevent PyTorch/OpenCV deadlocks on Windows
    cv2.setNumThreads(0)
    
    full_dataset = OCTDataset(dataframe=metadata_df, image_dir=data_dir, transform=oct_transform)
    # Using num_workers=0 to prevent windows deadlock
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    unique_classes = np.array([0, 1, 2, 3])
    outer_results = []
    
    print(f"\n{'='*50}\nEvaluating all 10 CNN Models on FULL Base Dataset\n{'='*50}")
    
    for outer_fold in range(10):
        print(f"\n--- Evaluating CNN Model Fold {outer_fold + 1}/10 ---")
        model_path = os.path.join(models_dir, f"cnn_fold_{outer_fold + 1}.pth")
        
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found! Skipping fold {outer_fold + 1}.")
            continue
            
        fold_model = OCTConvNet(out_c1=16, out_c2=32, dense_units=128).to(device)
        try:
            fold_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except TypeError:
            fold_model.load_state_dict(torch.load(model_path, map_location=device))
        
        preds, true_labels = predict_cnn(fold_model, full_loader, device)
        test_metrics = compute_classification_metrics(true_labels, preds, unique_classes)
        print(f"  -> Fold {outer_fold + 1} Base Test Metrics: {test_metrics}")
        outer_results.append(test_metrics)
        
    if not outer_results:
        print("No results to display. Make sure the models have been trained and saved.")
        return

    print(f"\n{'='*50}\nFinal Baseline Results on Clean Data (10 CNN Folds)\n{'='*50}")
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    for k in metrics_keys:
        values = [res[k] for res in outer_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{k.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()

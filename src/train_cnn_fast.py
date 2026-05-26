# imports
from pathlib import Path
from itertools import product
import os
import sys
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)
import torch
torch.set_num_threads(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# import custom cross validation function
src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cross_validation import stratified_kfold

# try to use gpu if available
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
print(f"Using device: {device}")

# set debug mode to run on a sample
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 10

# set up dataset (which can be used to point to the clean or degraded images)
class OCTDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]["filename"]
        label = self.df.iloc[idx]["label"]
        img_path = self.image_dir / filename
        image = cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback to pure noise to prevent crashing the entire nested CV
            return torch.randn(1, 224, 224), label
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label


# define transformation we apply
oct_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# define model (adapted from tut8)
OUT_C1 = 16
OUT_C2 = 32

class OCTConvNet(nn.Module):
    def __init__(self,out_c1, out_c2, dense_units, n_classes=4, img_size=224):
        super(OCTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_c1, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=out_c1, out_channels=out_c2, kernel_size=5)
        
        # auto-calc flatten size
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




# classification metrics (we want acc, recall, precision, f1)
def compute_classification_metrics(
    y_true,
    y_pred,
    unique_classes
):

    precisions = []
    recalls = []
    f1_scores = []

    for c in unique_classes:

        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0

    return (accuracy, np.mean(precisions), np.mean(recalls), np.mean(f1_scores))


# prediction
def predict_cnn(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


# training
def train_cnn_model(train_dataset, train_idx, val_dataset, val_idx, dense_units, lr, batch_size, max_epochs, patience, device):

    model = OCTConvNet( out_c1=OUT_C1, out_c2=OUT_C2, dense_units=dense_units).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)

    # only create val loader if exists
    if val_dataset is not None and val_idx is not None:
        val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # initialise starting values
    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # train the model
        model.train()
        running_train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # validaition
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f}")

            # early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


# metadata (image names, paths, labels etc)
import os
parent_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
metadata_df = pd.read_csv(parent_dir / "data/image_metadata.csv")


# if using debug sample
if DEBUG_MODE:
    metadata_df = metadata_df.sample(n=DEBUG_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"DEBUG MODE ENABLED: Using {len(metadata_df)} images")
else:
    # Match exactly the ViT 15% subset
    patient_map = {}
    patient_counter = 0
    mapped_groups = []
    for p in metadata_df["patient_id"].values:
        if p not in patient_map:
            patient_map[p] = patient_counter
            patient_counter += 1
        mapped_groups.append(patient_map[p])
    mapped_groups = np.array(mapped_groups)
    
    y_tmp = metadata_df["label"].values
    
    # Needs stratified_kfold to be imported, which it is globally at the top of the notebook
    subset_splits = stratified_kfold(metadata_df, y_tmp, groups=mapped_groups, n_splits=int(1.0/0.15), shuffle=True, random_state=42)
    _, subset_idx = subset_splits[0]
    
    metadata_df = metadata_df.iloc[subset_idx].reset_index(drop=True)
    print(f"SUBSET ENABLED: Using 15% of images ({len(metadata_df)}) matching ViT notebook.")

# set up the two datasets
clean_dataset = OCTDataset(dataframe=metadata_df, image_dir=parent_dir / "data/processed",
    transform=oct_transform)
degraded_dataset = OCTDataset(dataframe=metadata_df, image_dir=parent_dir / "data/degraded",
    transform=oct_transform)

# set up the labels we need
y = metadata_df["label"].values
groups = metadata_df["patient_id"].values
unique_classes = np.array([0, 1, 2, 3])

# hyper params for tuning
param_grid = {
    "lr": [5e-4, 1e-3, 5e-3],
    "batch_size": [16, 32, 64]
}

# grid search combos
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in product(*values)]


# outer cross val
outer_splits = stratified_kfold(metadata_df, y, groups=groups, n_splits=10, shuffle=True, random_state=42)

clean_results = []
degraded_results = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):

    print(f"\n======== Fold {fold_idx + 1} ==========")

    best_params = {"lr": 0.005, "batch_size": 16}
    print(f"Using fixed best params: {best_params}")

    # FINAL train
    final_model = train_cnn_model(train_dataset=clean_dataset,
        train_idx=train_idx, val_dataset=None,
        val_idx=None, dense_units=128, lr=best_params["lr"],
        batch_size=best_params["batch_size"], max_epochs=20, patience=None, device=device)


    # Save final model for this fold
    os.makedirs(str(parent_dir / 'models'), exist_ok=True)
    torch.save(final_model.state_dict(), str(parent_dir / f'models/cnn_fold_{fold_idx + 1}.pth'))

    # clean eval
    clean_loader = DataLoader(Subset(clean_dataset, test_idx), batch_size=best_params["batch_size"], shuffle=False)
    clean_preds = predict_cnn(final_model, clean_loader, device)
    y_test = y[test_idx]
    clean_metrics = compute_classification_metrics(y_test, clean_preds, unique_classes)
    clean_results.append(clean_metrics)

    # degraded eval
    degraded_loader = DataLoader(Subset(degraded_dataset, test_idx), batch_size=best_params["batch_size"], shuffle=False)
    degraded_preds = predict_cnn(final_model, degraded_loader, device)
    degraded_y = metadata_df.iloc[test_idx]["label"].values
    degraded_metrics = compute_classification_metrics(degraded_y, degraded_preds, unique_classes)
    degraded_results.append(degraded_metrics)

    # results from this fold
    print(f"Fold {fold_idx + 1} | Clean Acc: {clean_metrics[0]:.4f} | Clean F1: {clean_metrics[3]:.4f} | Degraded F1: {degraded_metrics[3]:.4f}")

# calc final results
clean_results = np.array(clean_results)
degraded_results = np.array(degraded_results)

metric_names = ["Accuracy", "Precision", "Recall", "F1"]

# celan results
print("\n======= CLEAN RESULTS =========")

clean_final_stats = {}
for i, metric_name in enumerate(metric_names):
    mean_val = np.mean(clean_results[:, i])
    std_val = np.std(clean_results[:, i])
    clean_final_stats[metric_name] = {
        "mean": mean_val,
        "std": std_val
    }
    # print metric w standard error
    print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

# degraded results
print("\n======== DEGRADED RESULTS =========")

degraded_final_stats = {}
for i, metric_name in enumerate(metric_names):
    mean_val = np.mean(degraded_results[:, i])
    std_val = np.std(degraded_results[:, i])
    degraded_final_stats[metric_name] = {
        "mean": mean_val,
        "std": std_val
    }
    # print metric w standard error
    print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")


def plot_metrics_with_error_bars(final_stats, title):
    metrics = list(final_stats.keys())
    means = [final_stats[m]["mean"] for m in metrics]
    stds = [final_stats[m]["std"] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, means, yerr=stds, capsize=10, edgecolor="black")
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis="y",linestyle="--",alpha=0.7)
    plt.show()

plot_metrics_with_error_bars(clean_final_stats, "Clean OCT Performance 10-Fold Nested CV")

plot_metrics_with_error_bars(degraded_final_stats, "Degraded OCT Performance 10-Fold Nested CV")
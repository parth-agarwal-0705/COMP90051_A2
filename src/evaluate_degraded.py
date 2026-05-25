import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import csv

# Add the parent directory to the path so it can be run from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cross_validation import stratified_kfold

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=256, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class LightweightViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4, 
                 embed_dim=256, depth=4, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits

class OCTDataset(Dataset):
    def __init__(self, image_paths, labels, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        if not os.path.exists(path):
            img = torch.randn(3, self.img_size, self.img_size)
            return img, torch.tensor(label, dtype=torch.long)
        
        try:
            img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_np is None: raise ValueError('Image not found')
            img_np = cv2.resize(img_np, (self.img_size, self.img_size))
            img_np = np.stack((img_np,)*3, axis=0)
            img = torch.tensor(img_np, dtype=torch.float32) / 255.0
        except Exception as e:
            img = torch.randn(3, self.img_size, self.img_size)
            
        return img, torch.tensor(label, dtype=torch.long)

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    accuracy = np.mean(y_true == y_pred)
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        'accuracy': accuracy,
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return calculate_metrics(all_labels, all_preds)

def load_dataset_metadata(data_dir, csv_path):
    image_paths, labels, groups = [], [], []
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    patient_counter = 0
    patient_map = {}
    
    if not os.path.exists(csv_path) or not os.path.exists(data_dir):
        print(f"Warning: {csv_path} or {data_dir} does not exist.")
        return np.array([]), np.array([]), np.array([])
        
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try: header = next(reader)
        except StopIteration: pass
        
        for row in reader:
            if len(row) < 6: continue
            
            disease = row[0]
            label_str = row[1]
            patient_id_str = row[2]
            img_name = row[4]
            
            if label_str.upper() in class_to_idx:
                label_idx = class_to_idx[label_str.upper()]
            elif disease.upper() in class_to_idx:
                label_idx = class_to_idx[disease.upper()]
            else:
                try: label_idx = int(label_str)
                except ValueError: continue
                
            img_path = os.path.join(data_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            if patient_id_str not in patient_map:
                patient_map[patient_id_str] = patient_counter
                patient_counter += 1
            patient_id = patient_map[patient_id_str]
            
            image_paths.append(img_path)
            labels.append(label_idx)
            groups.append(patient_id)
            
    return np.array(image_paths), np.array(labels), np.array(groups)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Base directories (assumes running from project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "degraded")
    csv_path = os.path.join(base_dir, "data", "image_metadata.csv")
    models_dir = os.path.join(base_dir, "models")
    
    X_paths, y_labels, groups = load_dataset_metadata(data_dir=data_dir, csv_path=csv_path)
    
    if len(X_paths) == 0:
        print("No images found. Check data paths.")
        return
        
    print(f"Total degraded images linked: {len(X_paths)}")
    
    print(f"\n{'='*50}\nEvaluating all 10 Models on FULL Degraded Dataset (Size: {len(X_paths)})\n{'='*50}\n")
    
    outer_results = []
    batch_size = 32
    
    cv2.setNumThreads(0)
    full_dataset = OCTDataset(X_paths, y_labels)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    for outer_fold in range(10):
        print(f"\n--- Evaluating Model Fold {outer_fold + 1}/10 ---")
        
        model_path = os.path.join(models_dir, f"vit_fold_{outer_fold + 1}.pth")
        
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found! Skipping fold {outer_fold + 1}.")
            continue
            
        fold_model = LightweightViT(num_classes=4).to(device)
        try:
            fold_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions without weights_only
            fold_model.load_state_dict(torch.load(model_path, map_location=device))
        
        test_metrics = evaluate_model(fold_model, full_loader, device)
        print(f"  -> Model Fold {outer_fold + 1} Degraded Full Test Metrics: {test_metrics}")
        outer_results.append(test_metrics)
        
    if not outer_results:
        print("No results to display. Check if models exist.")
        return

    print(f"\n{'='*50}\nFinal Robustness Results on Degraded Data (10 Folds)\n{'='*50}")
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    for k in metrics_keys:
        values = [res[k] for res in outer_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{k.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()

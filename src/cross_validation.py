import numpy as np

def stratified_kfold(X, y, groups, n_splits=10, shuffle=True, random_state=None):
    """
    Splits data into train and test indices ensuring that:
    No group (patient ID) is present in both train and test sets for any fold.
    """
    y = np.array(y)
    groups = np.array(groups)
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Map each unique group (patient) to its total class counts and its dominant class
    unique_groups = np.unique(groups)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(unique_groups)
        
    group_class_counts = {}
    group_labels = {}
    
    for g in unique_groups:
        group_mask = (groups == g)
        y_g = y[group_mask]
        
        # Count how many of each class this specific patient has
        counts = np.bincount(y_g, minlength=n_classes)
        group_class_counts[g] = counts
        # Identify the primary label for this group 
        group_labels[g] = np.argmax(counts)
        
    # Initialize empty buckets for tracking split contents
    fold_indices = [[] for _ in range(n_splits)]
    fold_class_counts = np.zeros((n_splits, n_classes))
    
    # Distribute groups class-by-class to maintain stratification proportions
    for c in unique_classes:
        # Pull all patients whose dominant class matches 'c'
        class_groups = [g for g in unique_groups if group_labels[g] == c]
        
        # Sort these patients by total scans descending to distribute heavy groups first
        class_groups = sorted(class_groups, key=lambda g: np.sum(group_class_counts[g]), reverse=True)
        
        for g in class_groups:
            g_counts = group_class_counts[g]
            
            # Find the specific fold that currently needs this class the most 
            best_fold = np.argmin(fold_class_counts[:, c])
            
            # Record all matching indices for this entire patient group into that fold
            actual_indices = np.where(groups == g)[0]
            fold_indices[best_fold].extend(actual_indices)
            
            # Update the class tracking state for that fold
            fold_class_counts[best_fold] += g_counts
            
    # Construct the standard output train/test tuples
    splits = []
    for test_fold_idx in range(n_splits):
        test_idx = np.array(fold_indices[test_fold_idx], dtype=int)
        
        # Combine all indices from other folds to form the training collection
        train_idx = np.concatenate([
            fold_indices[i] for i in range(n_splits) if i != test_fold_idx
        ], axis=0).astype(int)
        
        splits.append((train_idx, test_idx))
        
    return splits
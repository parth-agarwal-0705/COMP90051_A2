import numpy as np

def stratified_kfold(X, y, n_splits=10, shuffle=True, random_state=None):
    """
    Splits data into train and test indices from scratch while preserving class ratios.
    """
    y = np.array(y)
    n_samples = len(y)
    
    # Handle shuffling if requested
    if shuffle:
        rng = np.random.default_rng(random_state)
    
    # Group the original indices by their unique class labels
    unique_classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}
    
    # Shuffle indices within each class to ensure randomness across runs
    if shuffle:
        for c in class_indices:
            rng.shuffle(class_indices[c])
            
    # Create buckets for each fold
    folds = [[] for _ in range(n_splits)]
    
    # Distribute indices round-robin style across folds to preserve stratification
    for c in unique_classes:
        indices = class_indices[c]
        for i, idx in enumerate(indices):
            fold_idx = i % n_splits
            folds[fold_idx].append(idx)
            
    # Construct the final train/test index pairs for all splits
    splits = []
    for test_fold_idx in range(n_splits):
        test_indices = np.array(folds[test_fold_idx])
        
        # Merge all other folds to create the training set
        train_indices = np.concatenate([
            folds[i] for i in range(n_splits) if i != test_fold_idx
        ])
        
        splits.append((train_indices, test_indices))
        
    return splits
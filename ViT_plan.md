# Implementation Plan: Custom Vision Transformer (ViT)

## Objective
Implement a lightweight Vision Transformer (ViT) from scratch in PyTorch within `notebooks/Vision_transformer.ipynb`. The model will be trained and evaluated using a custom nested cross-validation pipeline to strictly adhere to the "from scratch" constraints of the COMP90051 project.

## Scope & Impact
* **Target File:** `notebooks/Vision_transformer.ipynb`
* **Dependencies:** Relies on existing custom scripts `src/preprocess.py` and `src/cross_validation.py`.
* **Constraint Adherence:** No third-party tools (like `scikit-learn`) will be used for cross-validation splitting, metric calculation, or hyperparameter grid search.

## Proposed Solution

### Phase 1: Custom ViT Architecture (PyTorch)
Implement the core components of a Vision Transformer from scratch to maintain control over model complexity for local training:
1.  **Patch Embedding Layer:** A `nn.Conv2d` module to slice the $224 \times 224$ preprocessed images into patches (e.g., $16 \times 16$) and project them into an embedding dimension.
2.  **Positional Encoding:** Add learnable positional embeddings to the patch sequence, including the `[CLS]` token.
3.  **Transformer Encoder Blocks:** Implement a sequence of standard transformer layers using `nn.TransformerEncoderLayer` or raw Multi-Head Attention + MLP blocks.
4.  **Classification Head:** An MLP head attached to the `[CLS]` token output to classify into the 4 clinical classes.

### Phase 2: Pipeline Initialization
1.  **Dataset Preparation:** Finalize the `OCTDataset` class (already partially implemented in the notebook) to load all image paths, apply `preprocess_image`, and yield PyTorch tensors.
2.  **Dummy Data Load:** Write code to load a manageable subset or the full dataset directory structure to extract file paths, labels, and group IDs (patient IDs) required for stratification.

### Phase 3: Nested Cross-Validation Implementation
Implement the evaluation framework completely from scratch:
1.  **Hyperparameter Grid:** Define a search space suitable for local training (e.g., `learning_rates = [1e-3, 1e-4]`, `batch_sizes = [16, 32]`).
2.  **Outer Loop ($K=10$):**
    *   Call `stratified_kfold` to partition the dataset into 10 training/testing splits based on patient IDs.
3.  **Inner Loop ($K=3$):**
    *   For each outer fold's training set, call `stratified_kfold` again with $K=3$ to create a validation split.
    *   Implement a nested `for` loop to iterate over the hyperparameter grid. Train the ViT model on the inner training set and evaluate on the inner validation set.
    *   Select the hyperparameter combination yielding the highest metric (e.g., Accuracy or Macro F1).

### Phase 4: Training & Evaluation
1.  **Final Model Training:** Train the ViT model using the optimal hyperparameters found in Phase 3 on the entire outer fold training dataset.
2.  **Custom Metrics:** Evaluate predictions on the outer fold test set using the custom `calculate_metrics` function. Store the results for all 10 folds.
3.  **Visualization:** Use `matplotlib` to plot the final performance metrics (Accuracy, Precision, Recall, F1) across the 10 folds, including calculated error bars (standard deviation).

## Alternatives Considered
*   **Pre-built `torchvision` ViT:** Initializing `torchvision.models.vit_b_16(weights=None)` was considered but rejected as the standard architecture is computationally heavy ($86M$ parameters) and might be too slow for complete local nested cross-validation without substantial computational resources.

## Verification
*   Verify that `sklearn` is completely absent from the implementation blocks.
*   Verify the training loop correctly updates weights and the loss decreases.
*   Verify that the resulting evaluation plots contain explicit error bars representing the variance across the 10 outer folds.
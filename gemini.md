Here is a concise context overview designed to quickly bring a code assistant up to speed on our project’s scope, constraints, and architecture:

---

## Project Overview: COMP90051 Group Project

### 1. Objective & Constraints

* 
**Goal:** Execute a rigorous machine learning research pipeline analyzing model performance and robustness under specific data conditions. Scientific process and manual implementation are prioritized over raw optimization metrics.


* **Strict "From-Scratch" Requirement:** Third-party libraries (e.g., `scikit-learn`) are **strictly prohibited** for the following pipeline components:
* Cross-validation splitting.


* Nested cross-validation for hyperparameter tuning.


* Experimental metric calculations (Accuracy, F1-score, Precision, Recall, etc.).




* 
**Environment:** Google Colab (with free GPU acceleration) integrated with a private GitHub repository for tracking commit history and team logs.



---

### 2. Dataset Specification

* **Dataset:** Kermany 2018 Retinal Optical Coherence Tomography (OCT2017).
* **Scale:** ~84,000 high-resolution grayscale images split across 4 clinical classes (CNV, DME, Drusen, Normal). This easily exceeds the syllabus constraint of $\ge 10,000$ images at $\ge 100 \times 100$ pixels.


* **Research Question:** Evaluating how model architectures of varying complexity degrade when subjected to simulated real-world clinical interference (specifically **Speckle Noise** and **Resolution Degradation**).

---

### 3. Pipeline & Architectural Blueprint

* **Data Engineering:** Manual implementation of Contrast-Limited Adaptive Histogram Equalization (CLAHE) for preprocessing, alongside custom intensity normalization.
* 
**Validation Strategy:** * **Outer Loop:** Custom **Stratified $K$-Fold Cross-Validation** ($K \ge 10$) written from scratch to preserve imbalanced class ratios across folds.


* 
**Inner Loop:** Custom **Nested Cross-Validation** ($K \ge 3$) for hyperparameter tuning to prevent data leakage.




* 
**Model Spectrum:** 1.  *Simple:* SVM or Random Forest utilizing manually extracted visual features (e.g., Local Binary Patterns or HOG).
2.  *Medium:* A standard Convolutional Neural Network (e.g., VGG16 or a shallow custom CNN).
3.  *Complex:* A modern, post-2016 framework (e.g., Vision Transformer / Mobile-ViT) referenced from an approved tier-1 conference (NeurIPS/ICLR/ICML).


* 
**Downstream Evaluation:** Custom metric scripts yielding final performance curves complete with calculated **error bars** across all 10 outer iterations.



---

### Instructions for Code Generation

> When writing code or helper scripts for this workspace, ensure all cross-validation splitters, parameter grid loops, and evaluation metrics are written using raw Python/NumPy logic. Do not import automated pipeline tools from `sklearn.model_selection` or `sklearn.metrics`. Deep learning frameworks (PyTorch/TensorFlow) may be leveraged solely for model instantiation and forward/backward passes.
> 
>
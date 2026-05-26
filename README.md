# COMP90051_A2

# Question: 
How Do Different Models Maintain Classification Metrics for OCT Image Scan Classification When Subjected to Simulated Artefacts?

# Team
Andy Chen 1452766
Lance Davine 1462196
Parth Agarwal 1460308

# Dataset:
Kermany, D., Zhang, K., Goldbaum, M. (2018). Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images  (Version 3) [Data set] Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.3

# How to run code:
Note: All code is run inside their respective notebooks. The given order must be followed. 
Approximately 30-50GB of disk space is requred.

1. Run 'setup_EDA.ipynb' (establishes directories and loads data)
2. Run 'processing.ipynb' (preprocessing to all images)
3. Run 'degrading.ipynb' (degradation to all images)

Then, each model can be run using their respective notebook:
- Run 'SVM.ipynb'
- Run 'CNN.ipynb'
- Run 'Vision_transformer.ipynb'

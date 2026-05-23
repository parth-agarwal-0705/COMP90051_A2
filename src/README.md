Run this in your models to use the code for stratified k-fold cross validation:

```python
import os
import sys
import numpy as np

src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cross_validation import stratified_kfold
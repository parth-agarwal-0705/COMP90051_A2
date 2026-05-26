# How to include these python scripts in the notebooks

Run this in your models to use the code for stratified k-fold cross validation:
```python
import os
import sys
import numpy as np

src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cross_validation import stratified_kfold
```


Run this in your models to use the code to preprocess an image:
```python
import os
import sys
import numpy as np

src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preprocess import preprocess_image, preprocess_image_array
```


Run this in your models to use the code to augment/corrupt/degrade an image:
```python
import os
import sys
import numpy as np

src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from degradation import apply_degradation, apply_degradation_and_preprocess
```

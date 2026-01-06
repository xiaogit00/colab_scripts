# What is this for: 
A folder for you to import your modules in Google Colab Environments. The modules in each project are stored like so:

```
project/
├── train.py
├── dataset.py
├── model.py
├── losses.py
├── utils.py
└── __init__.py
```

# How to use this:
1. Open a new Google Colab file
2. Add the following lines: 
```
!git clone https://github.com/xiaogit00/colab_scripts.git
%cd colab_scripts/projectName
!pip install -r requirements.txt
```
3. Add project root to PYTHONPATH
```
import sys
sys.path.append('/content/colab_scripts/projectName')
```


1. Use the imports for that project!
```
from project.model import build_model
from project.dataset import get_dataloader
from project.utils import seed_everything
```
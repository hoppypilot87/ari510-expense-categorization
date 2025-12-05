# Data Description

## Dataset Format

The competition uses transaction data with numerical features. All data files are provided in space-separated format without headers.

### Files Provided

- **training_data**: Training features (one sample per line)
- **training_label**: Training labels (integer category IDs)
- **testing_data**: Testing features (one sample per line)
- **testing_label**: Testing labels (used for scoring, not provided to participants)

### Data Format

- All features are numeric
- Labels are integers from 0 to 11, representing the 12 expense categories
- Data is provided in numpy-compatible space-separated format

### Label Mapping

```
0: Bills
1: DiningOut
2: Education
3: Entertainment
4: Groceries
5: Healthcare
6: Miscellaneous
7: PersonalCare
8: Shopping
9: Transport
10: Travel
11: Utilities
```

## Loading the Data

Your model will receive data as numpy arrays:

```python
import numpy as np

# Training data will be loaded like this:
X_train = np.genfromtxt('training_data')
y_train = np.genfromtxt('training_label')

# Testing data
X_test = np.genfromtxt('testing_data')
```

## Dataset Statistics

- Training samples: ~7,000 transactions
- Testing samples: ~3,000 transactions
- Features: Transaction amount and other derived features
- Classes: 12 expense categories
- Class distribution: Balanced across categories

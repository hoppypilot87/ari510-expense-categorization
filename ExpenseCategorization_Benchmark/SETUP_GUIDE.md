# Codabench Competition Setup Guide

## Complete Step-by-Step Instructions

### Overview

You now have a complete Codabench competition bundle ready to upload. Here's what was created:

```
ExpenseCategorization_Benchmark/
├── competition.yaml           # Competition configuration
├── ingestion_program.zip      # Runs participant code
├── scoring_program.zip        # Evaluates predictions
├── public_data.zip            # Training/testing data
├── starting_kit.zip           # Sample solution for participants
└── pages/                     # Competition documentation
    ├── overview.md
    ├── data.md
    ├── evaluation.md
    └── rules.md
```

---

## Step 1: Create a New Competition on Codabench

1. Go to https://www.codabench.org/
2. Log in to your account
3. Click **"Create Competition"** or **"New Benchmark"**
4. Choose **"Upload a competition bundle"**

---

## Step 2: Upload the Competition Configuration

### Option A: Upload competition.yaml

1. Upload the `competition.yaml` file
2. Codabench will parse it and create the competition structure
3. This is the easiest method

### Option B: Manual Configuration

If uploading YAML doesn't work, configure manually:

1. **Basic Information**:
   - Title: `Automated Expense Categorization Challenge`
   - Description: `Build a machine learning model to automatically categorize financial transactions`
   - Docker Image: `codalab/codalab-legacy:py39`

2. **Tasks**:
   - Create a task named "Expense Category Prediction"
   - Description: "Predict the category of financial transactions"

---

## Step 3: Upload Program Files

Upload these files in the competition editor:

### 3.1 Ingestion Program
- **File**: `ingestion_program.zip`
- **Purpose**: Loads data, runs participant code, generates predictions
- **Location**: Upload in "Ingestion Program" section

### 3.2 Scoring Program
- **File**: `scoring_program.zip`
- **Purpose**: Evaluates predictions and calculates metrics
- **Location**: Upload in "Scoring Program" section

---

## Step 4: Upload Data Files

### 4.1 Public Data (Input Data)
- **File**: `public_data.zip`
- **Contains**:
  - `training_data` (7,000 samples)
  - `training_label` (7,000 labels)
  - `testing_data` (3,000 samples)
  - `testing_label` (3,000 labels - for scoring)
  - `label_mapping.txt` (category reference)
- **Location**: Upload as "Public Data" or "Input Data"

---

## Step 5: Create Phases

Create at least one phase:

### Development Phase
- **Name**: Development Phase
- **Description**: Practice and develop your model
- **Start Date**: Set to current date or competition start
- **Max Submissions per Day**: 5
- **Max Total Submissions**: 100
- **Enable Public Leaderboard**: Yes

---

## Step 6: Configure Leaderboard

Set up the leaderboard columns (in order):

1. **Accuracy** (Primary metric)
   - Key: `accuracy`
   - Sort: Descending
   - Precision: 4 decimals

2. **F1 Score (Weighted)**
   - Key: `f1_weighted`
   - Sort: Descending
   - Precision: 4 decimals

3. **F1 Score (Macro)**
   - Key: `f1_macro`
   - Sort: Descending
   - Precision: 4 decimals

4. **Precision**
   - Key: `precision`
   - Sort: Descending
   - Precision: 4 decimals

5. **Recall**
   - Key: `recall`
   - Sort: Descending
   - Precision: 4 decimals

6. **Duration**
   - Key: `duration`
   - Sort: Ascending (lower is better)
   - Precision: 2 decimals

---

## Step 7: Add Documentation Pages

Upload the markdown files to the "Pages" section:

1. **Overview** (`overview.md`): Competition description
2. **Data** (`data.md`): Dataset information
3. **Evaluation** (`evaluation.md`): Metrics explanation
4. **Rules** (`rules.md`): Submission requirements

---

## Step 8: Provide Starting Kit (Optional)

Upload `starting_kit.zip` as a downloadable resource:
- This helps participants get started quickly
- Contains baseline model and README

---

## Step 9: Test the Competition

Before making it public, test it:

1. **Make a test submission**:
   - Use the provided `starting_kit/model.py`
   - Zip it with `requirements.txt`
   - Submit to your competition

2. **Check the submission**:
   - Status should go: Submitting → Running → Finished
   - Should NOT stay in "Preparing" status
   - Check logs for any errors

3. **Verify scores**:
   - Scores should appear on leaderboard
   - All metrics should be calculated
   - Baseline should get ~80-90% accuracy

---

## Step 10: Publish the Competition

Once testing is successful:

1. Set competition visibility to "Public" (if desired)
2. Share the competition URL
3. Monitor submissions and leaderboard

---

## Troubleshooting Common Issues

### Issue: Submission stuck in "Preparing"
**Solution**:
- Check Docker image is correct: `codalab/codalab-legacy:py39`
- Verify all zip files were uploaded correctly
- Check ingestion/scoring program metadata.yaml files

### Issue: "Could not find scores file"
**Solution**:
- Verify scoring_program.zip was uploaded
- Check that scoring.py writes to `/app/output/scores.json`
- Check ingestion program completed successfully

### Issue: ImportError in submission
**Solution**:
- Check requirements.txt includes all dependencies
- Verify Docker image has required packages
- Consider using a more complete Docker image

### Issue: Wrong data format
**Solution**:
- Verify data files are space-separated (not CSV)
- Check no headers in data files
- Ensure labels are integers (not strings)

---

## File Locations Reference

All files are in: `c:\Users\grant\OneDrive\Documents\MS AI\Machine Learning\ExpenseCategorization\ExpenseCategorization_Benchmark\`

### Ready to Upload:
- `competition.yaml`
- `ingestion_program.zip`
- `scoring_program.zip`
- `public_data.zip`
- `starting_kit.zip`
- `pages/overview.md`
- `pages/data.md`
- `pages/evaluation.md`
- `pages/rules.md`

---

## Quick Start Checklist

- [ ] Create new competition on Codabench
- [ ] Upload competition.yaml OR configure manually
- [ ] Upload ingestion_program.zip
- [ ] Upload scoring_program.zip
- [ ] Upload public_data.zip
- [ ] Create Development Phase
- [ ] Configure leaderboard metrics
- [ ] Add documentation pages
- [ ] Upload starting_kit.zip
- [ ] Make test submission
- [ ] Verify scores appear correctly
- [ ] Publish competition

---

## Support

If you encounter issues:
1. Check Codabench documentation: https://github.com/codalab/codabench/wiki
2. Check submission logs for error messages
3. Verify all files match the expected format
4. Test locally before uploading

Good luck with your competition!

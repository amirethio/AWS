# 🧠 Full Educational Guide to Building a Prediction AI Model

## 📖 Introduction

Machine learning (ML) is the science of teaching computers to **learn patterns from data** and make decisions **without being explicitly programmed**. In healthcare, ML models can help detect diseases, predict outcomes, and assist doctors with faster diagnosis.

---

### 🧬 What is Machine Learning?

Machine Learning is a subfield of Artificial Intelligence (AI) that involves using algorithms to identify patterns in data. There are **three main types**:

| Type                | Description                              | Example                              |
|---------------------|------------------------------------------|--------------------------------------|
| Supervised Learning | Trains on labeled data (`X → y`)         | Predicting diabetes from patient records |
| Unsupervised Learning | Finds patterns in **unlabeled** data    | Grouping similar patients (clustering) |
| Reinforcement Learning | Learns via trial and error             | Training a robot to walk             |

---

### 🛠️ What Is Needed to Build an ML Model?

To train a machine learning model, we need:

- ✅ **Data** — like patient medical records  
- ✅ **A goal** — like predicting diabetes  
- ✅ **A model/algorithm** — built using mathematical functions, applied to your data (e.g., Logistic Regression, XGBoost)  
- ✅ **Evaluation metrics** — to check model performance (accuracy, precision, recall, etc.)  
- ✅ **Infrastructure** — to train, evaluate, and deploy the model (locally or on the cloud)

---

### 🔁 Traditional ML Workflow (Without AWS)

```text
Collect Data → Clean Data → Split Data → Train Model → Evaluate → Deploy → Monitor
```

---

### ☁️ How AWS SageMaker Makes This Easier

Amazon SageMaker simplifies and automates much of this process:

| Traditional Method           | With SageMaker                      |
|------------------------------|--------------------------------------|
| Manual data download         | Use S3 buckets                       |
| Jupyter on local computer    | Managed cloud-hosted Jupyter notebook |
| Manual deployment            | One-click deployment with endpoints  |
| Manual monitoring            | SageMaker Model Monitor              |

---

### 💻 What Is a Jupyter Notebook?

Let’s now start training our model using Amazon SageMaker.

On AWS, you **create a SageMaker notebook instance**, and it will open a **JupyterLab interface** in your browser, pre-installed with everything needed (Python, pandas, scikit-learn, boto3, etc.).

A **Jupyter Notebook** is an interactive coding environment used for writing code, visualizing data, and explaining what you’re doing — all in one document.

You can:
- Run and modify Python code  
- Visualize data inline (charts, tables)  
- Write notes beside your code  

---

### 🔧 Tools and Libraries We'll Use

| Tool         | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| `boto3`      | AWS SDK for Python — lets us download/upload files from S3             |
| `pandas`     | Python library for working with tabular data (e.g., loading `.csv`)    |
| `matplotlib` | For plotting and visualizing feature distributions                     |
| `scikit-learn` | For splitting datasets, training models, and evaluating performance |

---

## 📦 Dataset Columns

We'll use the **Pima Indian Diabetes Dataset**. Each row represents a female patient and includes medical measurements.

### ➤ Feature Columns:
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`

---

## 📁 Step 1: Download the Dataset from Amazon S3

Our dataset is stored in S3 (e.g., `training-bucket`), AWS’s cloud-based object storage service.  
Let’s go through the necessary steps to fetch and load the dataset.

---

### ✅ 1.1: Import Required Libraries

We import the essential Python libraries we'll use:

```python
import boto3          # AWS SDK to interact with S3
import pandas as pd   # For reading and manipulating tabular data
```

---

### ✅ 1.2: Define S3 File Info

Specify the S3 bucket name and the dataset file:

```python
bucket_name = 'training-bucket'     # Replace with your actual S3 bucket name
file_key = 'dataset.csv'            # File name inside S3
local_file = 'dataset.csv'          # Name to save locally
```

---

### ✅ 1.3: Download the File from S3

We use `boto3` to download the file from your bucket:

```python
s3 = boto3.client('s3')                         # Initialize the S3 client
s3.download_file(bucket_name, file_key, local_file)  # Download the CSV
```

---

### ✅ 1.4: Load and Preview the Dataset

We load the CSV file with `pandas`, rename the columns to meaningful names, and view the first few rows:

```python
df = pd.read_csv(local_file)

df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
              'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df.head()   # Display the first 5 rows
```

## 🧹 Step 2: Explore and Clean the Dataset

Now that we've loaded the dataset, we begin **data exploration and preprocessing**. This is a critical step to ensure the model is trained on meaningful and clean data.

---

### ✅ 2.1: Summary Statistics

Use `.describe()` to understand basic statistics like min, max, mean, and percentiles.

```python
df.describe()
```

This helps you:
- Detect strange values (like zeros in places where it doesn’t make sense)
- Get a quick sense of distribution and scale

---

### ✅ 2.2: Check for Invalid or Missing Values

Some features (like `Glucose`, `BMI`, `Insulin`) should never be zero. While they are not technically "missing" (`NaN`), zero is an **invalid** value in these medical contexts.

We’ll manually check these columns:

```python
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in invalid_zero_cols:
    count = (df[col] == 0).sum()
    print(f"{col}: {count} invalid (zero) entries")
```

---

### ✅ 2.3: Options to Handle Invalid Zeros

You have different choices to deal with invalid zero entries:

- ❌ **Remove Rows** — Drop rows with zeros (can reduce dataset size, risky if too many rows affected)
- ✅ **Replace with Mean** — Simple, but sensitive to outliers
- ✅ **Replace with Median** — More robust and common in medical datasets

In this project, we'll **replace zeros with the median** of non-zero values:

```python
for col in invalid_zero_cols:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)
```

---

### ✅ 2.4: Verify After Cleaning

Use `.describe()` again to confirm that the invalid zeros are gone:

```python
df.describe()
```

You should see updated min values for each cleaned column.

---

### ✅ 2.5: Plot Feature Distributions

Visualize all columns to understand the shape of their distributions:

```python
import matplotlib.pyplot as plt

df.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions After Cleaning", fontsize=16)
plt.tight_layout()
plt.show()
```

This helps you:
- Detect skewed features
- Check if the dataset has class imbalance
- Spot outliers and anomalies

---

### 🧾 Note on Class Balance

Our target variable `Outcome` is binary:  
- `0` = Not diabetic  
- `1` = Diabetic  

If one class dominates, the model may become biased and simply predict the dominant class.

You can check the class balance with:

```python
df['Outcome'].value_counts()
```

If the classes are **imbalanced**, some solutions include:
- Resampling the dataset (undersample or oversample)
- Using advanced techniques like SMOTE
- Applying class weights during model training

For now, we’ll just observe the distribution and continue.

---

### ✅ 2.6: Move Target Column to the Front (for SageMaker XGBoost)

SageMaker’s built-in XGBoost requires the **label (Outcome)** column to be the **first column**.

```python
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('Outcome')))
df = df[cols]
df.head()
```

---

✅ Done!  
We’ve now cleaned our dataset, replaced invalid values, verified the data, and reshaped it for training. you can go futher and clean the data to get better result for now let's pass to the next step

## ✂️ Step 3: Split and Prepare the Dataset for Training

Now that our dataset is clean and structured correctly, the next step is to **split it** into a **training set** and a **validation (testing) set**.

---

### 🧠 Why Split the Data?

We need to evaluate how well our model performs on **new, unseen data**.  
So we divide the dataset:

- **Training Set (80%)** — Used to teach the model
- **Validation Set (20%)** — Used to test the model’s generalization

This helps prevent **overfitting**, where the model memorizes the training data but fails on real-world data.

---

### ✅ 3.1: Import and Split the Dataset

We’ll use `train_test_split` from `scikit-learn` and stratify by `Outcome` to keep the class ratio balanced.

```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['Outcome']  # preserve class distribution
)
```

---

### ✅ 3.2: Save the Datasets as CSV (Without Header or Index)

SageMaker's XGBoost algorithm expects **raw CSV files** — no headers, no indexes.

```python
train_df.to_csv('train_final.csv', index=False, header=False)
val_df.to_csv('validation_final.csv', index=False, header=False)
```

These two files will be uploaded to Amazon S3 in the next step.

---

## ☁️ Step 4: Upload Training & Validation Files to Amazon S3

SageMaker does **not use local files directly**. Instead, it reads training data from **S3**, which is Amazon’s cloud storage.

In this step, we’ll upload:
- `train_final.csv` → `s3://your-bucket/train/train.csv`
- `validation_final.csv` → `s3://your-bucket/test/test.csv`

---

### 🧠 Why Use Amazon S3?

S3 is a **highly scalable**, **durable** cloud storage. SageMaker needs your dataset in S3 to launch a training job because:
- It can access data from anywhere in the cloud
- It separates data and compute layers for flexibility

---

### ✅ 4.1: Upload Files to S3

We’ll use `boto3.resource('s3')` to upload both files.

```python
import boto3

# S3 bucket and file keys
bucket_name = 'evangadi-ai'  
train_key = 'train/train.csv'
val_key = 'test/test.csv'

s3 = boto3.resource('s3')
s3.Bucket(bucket_name).upload_file('train_final.csv', train_key)
s3.Bucket(bucket_name).upload_file('validation_final.csv', val_key)

print("✅ Files uploaded successfully to S3.")
```

Make sure:
- Your bucket exists
- Your IAM role has permission to access S3
- File paths are accurate

---

## 🤖 Step 5: Define the Model and Start Training with SageMaker

Now that the data is in Amazon S3, we’ll set up **SageMaker to train a machine learning model**.  
We will use **XGBoost**, a powerful and popular algorithm for classification tasks like this one.

---

### 🧠 What is XGBoost?

XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting algorithm:
- Builds an **ensemble of decision trees**
- Learns from **mistakes** made by previous trees
- Excellent for **tabular data**, like CSVs
- Often ranks among **top performers** in real-world tasks and competitions

---

### ✅ 5.1: Define the XGBoost Estimator

To train a model in SageMaker, you create an **Estimator**, which describes:
- Which algorithm/container to use
- How many training machines (instances)
- Where to output the model
- Your IAM role

to find the container link for your specific region run this and 
```python

from sagemaker import image_uris
image_uris.retrieve(framework='xgboost',region='your_current_region eg()us-east-1',version='your_specific_version')

```

```python
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import get_execution_role

role = get_execution_role()  # Automatically gets the role attached to the notebook

container = '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest'

# SageMaker session and S3 location to store the output model
sagemaker_session = sagemaker.Session()
output_path = 's3://evangadi-ai/model-output'

# Define the estimator
estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',  # Use smaller type like 'ml.t2.medium' for testing
    output_path=output_path,
    sagemaker_session=sagemaker_session,
    base_job_name='evangadi-diabetes'
)
```

---

### ✅ 5.2: Set Hyperparameters

Hyperparameters are settings that control the **learning behavior** of the model.

```python

estimator.set_hyperparameters(
    max_depth=5,
    eta=0.1,
    subsample=0.7,
    objective='binary:logistic',
    num_round=150,
    scale_pos_weight=1.5,  # example ratio, calculate from your data
    early_stopping_rounds=10
)


```

> 🔍 Note: You can tune these values later to improve performance.

---

### ✅ 5.3: Point to Training and Validation Data

SageMaker needs to know **where your data lives in S3**.

```python
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(
    s3_data='s3://evangadi-ai/train/train.csv',
    content_type='csv'
)

val_input = TrainingInput(
    s3_data='s3://evangadi-ai/test/test.csv',
    content_type='csv'
)
```

---

### ✅ 5.4: Launch the Training Job

Now we call `.fit()` to start the job.

```python
estimator.fit({'train': train_input, 'validation': val_input})
```


---
##  Deployment Overview 

In this step, we’ve already trained our model on Amazon SageMaker.

➡ While the easiest deployment path is using **AWS endpoints**,  so you can simply use this sevice and you can get api endpoints for your model
but  now  for demonstration and learning purposes, we’ll show how to deploy the model on **EC2 or locally**. or on other services

>  Note: This requires building an API (e.g., FastAPI), and downloading the trained model from s3 

---


###  Step 6.1: Download the Trained Model from Amazon S3

After training the model in SageMaker, the trained model file is stored in your S3 bucket under the output path you specified.

When you use SageMaker's built-in XGBoost algorithm, the model is saved in a binary format named simply `model` (no file extension).  
To use this model locally or on an EC2 instance, we need to download it using the `boto3` AWS SDK.

In this step, we:
- Specify the S3 bucket name and model path
- Use `boto3.client('s3')` to connect to S3
- Download the model file and save it locally as `xgboost-model`

#### 📌 Example:
If your output path was:
`s3://evangadi-ai/model-output/evangadi-diabetes-2025-07-09-12-00-00/output/model`  
Then `bucket_name` is `evangadi-ai`, and `model_key` is the rest of the path after the bucket name.


### 🗂️ Step 6.2.1: Organize Folder and Set Up Virtual Environment

Before loading and running the model, it's a good practice to:
-  Create a new clean folder for your AI app
-  Set up a virtual environment inside it
-  Install only the dependencies you need (`boto3`, `xgboost`, `scikit-learn`, etc.)
-  Place the model file (`model.tar.gz`) inside the folder

---

### 📁 Recommended Project Structure

```
prediction-ai/
├── venv/                 ← Virtual environment (auto-created)
├── xgboost-model         ← Trained model file (from S3)
├── predictor.py          ← Python script to load and predict
└── requirements.txt      ← Optional: for saving dependencies
```

---

### 🛠️ Commands to Set Up

Run these in your terminal (Linux/Mac) or PowerShell (Windows):

```bash
# Create a new folder and move into it
mkdir prediction-ai # your project folder
cd prediction-ai

#  Create a Python virtual environment named 'venv'
python3 -m venv venv

#  Activate the virtual environment
# On Linux or macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

#  Install required libraries
pip install xgboost scikit-learn pandas boto3

```

---

###  Notes:
- Always **activate the virtual environment** before running your script.
- This keeps your project isolated and clean from global Python packages.


Once this setup is complete, move your `model.tar.gz` file into this folder.



###  Step 6.2.2: Extract the `.tar.gz` Model File

When SageMaker finishes training, it saves your model as a file named `model.tar.gz` in the `output/` directory on S3.

So when you download the trained model, you actually get a **compressed archive**.  
This archive usually contains a file simply called `model` (no extension), you will load with pickle or xgboost based on the container verison you have used for now we will use pickle

---

###  What We'll Do:
1. Make sure you’ve downloaded `model.tar.gz` from S3  
2. Extract it using Python or the terminal  
3. Get the inner `model` file ready to load

---

###  Option 1: Extract Using Python

Paste this in your script or a separate cell:

```python
import tarfile

# Replace with the actual filename if different
with tarfile.open("model.tar.gz", "r:gz") as tar:
    tar.extractall()  # Extracts files to the current directory

print("✅ model.tar.gz extracted successfully.")
```

This will extract the file named `model` into the same folder.

---

###  Option 2: Extract Using Terminal (Linux/macOS)

```bash
tar -xvzf model.tar.gz
```

> 📁 After extraction, you'll see a file called `model`. That’s the trained model file to load with `pickle`.

---

###  Reminder:

- If you're following our virtual environment setup (`prediction-ai/` folder), place `model.tar.gz` there and extract it inside the same folder.

---



## Step 6.3: Load Test Dataset for Prediction

---

### Conceptual Explanation

After training and extracting your model, the next task is to **load the test dataset** you want to evaluate. The test dataset is usually stored as a CSV file without headers because SageMaker XGBoost expects raw CSV files.

- The **first column is the label** (`Outcome`)—whether the patient is diabetic or not.
- The remaining columns are the features, in the exact order used during training.
- You must load this data carefully, explicitly naming the columns so the data matches the expected structure.

---
in this stape you can create a jupiter notebook file or python file to excute the next commads 
### Code 

```python
import pandas as pd

# Define the column names matching training dataset features + label
columns = ['Outcome', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Load test data CSV without headers, assigning column names explicitly
data = pd.read_csv("test.csv", header=None, names=columns)

print("✅ Test data loaded.")

# Separate features (X_test) and labels (y_test)
X_test = data.drop('Outcome', axis=1)
y_test = data['Outcome']
```

## Step 6.4: Load the Trained Model and Predict on Test Data


### ✔️ Built‑In SageMaker XGBoost (v1.3‑1 and Later)
- Models are saved in **native XGBoost binary format** using `Booster.save_model()`.
- To load locally:

  ```python
  import xgboost as xgb
  model = xgb.Booster()
  model.load_model("xgboost-model")
  ```
### 0lder or custom/script-based XGBoost (v1.2 or earlier, or training inside framework container)
Model may be saved using Python pickle.dump()

To load, use:

```python 

import pickle
model = pickle.load(open("xgboost-model", "rb"))

```
### 📄 Loading and Evaluating Pickled XGBoost Model Locally

This example shows how to load an XGBoost model saved using Python's `pickle` format, prepare test data, make predictions, and evaluate accuracy with a confusion matrix:

```python
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

# Load model with pickle (SageMaker Pickle format)
with open("model/xgboost-model", "rb") as f:
    model = pickle.load(f)
print(" Model loaded via pickle.")

# Convert features to numpy array (to avoid feature name mismatch)
X_test_array = X_test.to_numpy()

# Prepare DMatrix for prediction (no feature names)
dtest = xgb.DMatrix(X_test_array)

# Predict probabilities of the positive class (diabetic)
y_prob = model.predict(dtest)

# Convert probabilities to binary predictions using threshold 0.5
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Output some sample predictions
for i in range(5):
    print(f"Patient {i+1}: Probability={y_prob[i]:.2f}, Predicted={y_pred[i]}, Actual={y_test.iloc[i]}")

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy on test set: {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```


After you have tested the model file, you can choose how to build an API for your application. One popular option is to use **FastAPI**, which is a modern, fast (high-performance) web framework for building APIs with Python. 

Once you build your API (the backend), you can deploy it on an AWS EC2 instance. For a complete guide on how to deploy a full-stack application on an EC2 instance, including backend and frontend, you can refer the document build while depolying Evanagdi-Forum on single EC2 instance:

[https://github.com/amirethio/AWS/blob/main/Full-stuck-app-on-EC2.md]

If you find any mistakes in this documentation or have suggestions or useful additions, please feel free to contact me . We can update and improve this documentation together to make it more accurate and helpful. Your feedback is always welcome!










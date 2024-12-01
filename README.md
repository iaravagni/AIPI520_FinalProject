# Repeat Buyers Prediction Challenge
More details about the competition can be found here: [link](https://tianchi.aliyun.com/competition/entrance/231576)

## Team Members:
- Sakshee Patil
- Iara Ravagni

## Project Overview
This project focuses on predicting whether new buyers will become loyal, repeat customers for merchants on the "Double 11" sales day. Given a set of merchants and their new buyers during the promotion, our task is to predict which users are likely to make repeat purchases within 6 months. By targeting potential repeat buyers, merchants can optimize their promotional efforts and enhance their return on investment (ROI).

## Dataset
The dataset provided for this challenge includes anonymized user shopping logs for the 6 months before and on the "Double 11" day, with the goal of predicting if users will become repeat buyers. The training data contains labeled records of users, while the test data provides user-merchant pairs to predict. Key data features include:

- **User Profile**: Age range, gender
- **Merchant and User Interaction Logs**: Interaction records, including item, category, brand, and action type (click, add-to-cart, purchase, favorite)
- **Target Label**: Indicates whether the user is a repeat buyer for the given merchant (1 for repeat, 0 for non-repeat)

## Problem Definition
Merchants often run large promotional events to attract new customers, but many of these customers make only one-time purchases. By predicting the likelihood of repeat purchases, merchants can more effectively target customers with long-term value. Our goal is to predict the probability that new buyers will return to the same merchant within 6 months.

## Data Pipeline:
We developed a comprehensive data pipeline to clean, preprocess, and engineer features from the raw dataset. This pipeline includes:
1. **Data Cleaning**: Handling missing values by imputing with mean.
2. **Feature Engineering**: Extensive feature engineering was performed to extract user-based features and merchant-based features such as behavioural patterns, diversity patterns, abandonment rates, category and item preferences etc.
3. **Scaling**: Standardizing the data using `StandardScaler` to ensure uniformity across all features.
4. **Feature Selection**: Features were selected based on co-relation with the predictor variable `label`.
5. **Data Splitting**: Dividing the data into training, validation, and testing datasets, ensuring a stratified split to maintain balance between the classes.
6. **Hyperparameter Tuning**: Hyperparameters were tuned using the hyperopt library.
7. **Cross-Validation**: Used cross-validation to ensure the models are not overfitting and are generalized well across different data subsets.

## Models:
1. **Non-Deep Learning Model (Ensemble of Logistic Regression, Random Forest and XGBoost)**:
   - ...
   - Evaluation Metric: AUROC (Area Under Receiver Operating Characteristic Curve).

2. **Deep Learning Model (Neural Network)**:
   - A fully connected feedforward neural network was trained from scratch using PyTorch. ....
   - We experimented with different architectures and hyperparameters (e.g., learning rate, number of layers, early stopping, optimizers like SGD and Adam) to optimize the model's performance.
   - Evaluation Metric: AUROC.

## Evaluation Strategy:
- We used **AUROC** as the primary evaluation metric, which is ideal for imbalanced classification problems. AUROC measures the ability of the model to distinguish between the positive and negative classes, regardless of the threshold.
- By using AUROC, we ensure that merchants can set thresholds that align with their specific business goals. For instance, prioritizing high precision to reduce unnecessary costs or maximizing recall to identify as many potential repeat buyers as possible. This flexibility makes AUROC an ideal choice for guiding decision-making in a real-world application.
- We split the data into a training set (80%) and validation set (20%). We also evaluated the models on a separate test set to check their generalization.


## Running the Project:
To run the project and train the models, follow these steps:

### 1. Download the Dataset
The dataset is too large to be hosted in this repository and needs to be downloaded locally. 

1. Visit [Tianchi Competition Page](https://tianchi.aliyun.com/competition/entrance/231576) to download the dataset.
2. Unzip the downloaded dataset (`data_format1.zip`) and place it in the working directory under a folder named `data_format1`.

### 2. Set Up the Environment
It is recommended to use a virtual environment to manage dependencies and avoid conflicts. To set up and activate the environment:

```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate
```
### 3. Install Dependencies:
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 4. Usage
```bash
python pipeline.py
```

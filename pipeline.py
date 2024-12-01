import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Seeding
import random
random.seed(42)  # Set random seed for Python
np.random.seed(42)  # Set random seed for NumPy

import non_nn_models
import neural_network


def get_data():
    """
    Load and return the training, test, and supplementary datasets.

    Returns:
        tuple: DataFrames for train, user_info, user_log, and test datasets.
    """

    train = pd.read_csv('data_format1/train_format1.csv')
    user_info = pd.read_csv('data_format1/user_info_format1.csv')
    user_log = pd.read_csv('data_format1/user_log_format1.csv')
    test = pd.read_csv('data_format1/test_format1.csv')

    return train, user_info, user_log, test

def time_based_feature_eng(user_log):
    """
    Generate time-based features from user log data.
    Returns:
        DataFrame: Aggregated time-based features at the user level.
    """

    # Aggregating time-based features
    time_features = user_log.groupby('user_id').agg({
        'day_of_week': lambda x: x.value_counts().idxmax(),  # Most common interaction day
        'month': lambda x: x.value_counts().idxmax()         # Most active month
    }).reset_index()

    # Rename columns for clarity
    time_features.columns = ['user_id', 'most_active_day', 'most_active_month']  

    return time_features 


def user_based_feature_eng(user_log, time_features):
    """
    Generate user-based features from user log data, incorporating time-based features.
    Returns:
        DataFrame: Aggregated user-based features.
    """

    user_features = user_log.groupby('user_id').agg({
        'item_id': 'count',       # Total interactions
        'cat_id': 'nunique',      # Unique categories
        'seller_id': 'nunique',   # Unique merchants
        'brand_id': 'nunique',    # Unique brands
        'time_stamp': 'nunique',  # Active days
    }).reset_index()

    # Rename columns for clarity
    user_features.rename(columns={
        'item_id': 'total_interactions',
        'cat_id': 'unique_categories',
        'seller_id': 'unique_merchants',
        'brand_id': 'unique_brands',
        'time_stamp': 'active_days'
    }, inplace=True)

    user_features = user_features.merge(time_features, how='left', on='user_id')

    # Add action type-specific counts
    action_counts = user_log.groupby('user_id')['action_type'].value_counts().unstack(fill_value=0).reset_index()
    action_counts.columns = ['user_id', 'total_clicks', 'total_carts', 'total_purchases', 'total_favs']

    # Merge action counts with general features
    user_features = user_features.merge(action_counts, on='user_id', how='left')

    # Add ratio and conversion features
    user_features['total_actions'] = user_features[['total_clicks', 'total_carts', 'total_purchases', 'total_favs']].sum(axis=1)
    user_features['click_ratio'] = user_features['total_clicks'] / user_features['total_actions']
    user_features['purchase_ratio'] = user_features['total_purchases'] / user_features['total_actions']
    user_features['add_to_cart_ratio'] = user_features['total_carts'] / user_features['total_actions']
    user_features['favorite_ratio'] = user_features['total_favs'] / user_features['total_actions']

    # Direct abandonment rates
    user_features['cart_abandonment_rate'] = np.where(
        user_features['total_carts'] > 0,
        (user_features['total_carts'] - user_features['total_purchases']) / user_features['total_carts'],
        0
    )

    user_features['fav_abandonment_rate'] = np.where(
        user_features['total_favs'] > 0,
        (user_features['total_favs'] - user_features['total_purchases']) / user_features['total_favs'],
        0
    )

    # Weighted abandonment rates (relative to total actions)
    user_features['weighted_cart_abandonment_rate'] = (
        (user_features['total_carts'] - user_features['total_purchases']) / user_features['total_actions']
    )

    user_features['weighted_fav_abandonment_rate'] = (
        (user_features['total_favs'] - user_features['total_purchases']) / user_features['total_actions']
    )

    # Diversity features
    user_features['category_preference'] = user_log.groupby('user_id')['cat_id'].apply(lambda x: x.value_counts(normalize=True).max()).values
    user_features['merchant_preference'] = user_log.groupby('user_id')['seller_id'].apply(lambda x: x.value_counts(normalize=True).max()).values
    user_features['brand_preference'] = user_log.groupby('user_id')['brand_id'].apply(lambda x: x.value_counts(normalize=True).max()).values

    # Behavioral patterns
    user_features['action_entropy'] = user_log.groupby('user_id')['action_type'].apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True))))
    ).values

    # Repeat behavior features
    user_features['repeat_category_ratio'] = user_log.groupby('user_id')['cat_id'].apply(
        lambda x: (x.value_counts() > 1).sum() / len(x)
    ).values

    user_features['repeat_brand_ratio'] = user_log.groupby('user_id')['brand_id'].apply(
        lambda x: (x.value_counts() > 1).sum() / len(x)
    ).values

    user_features['repeat_merchant_ratio'] = user_log.groupby('user_id')['seller_id'].apply(
        lambda x: (x.value_counts() > 1).sum() / len(x)
    ).values

    return user_features


def merchant_based_feature_eng(user_log):
    """
    Generate merchant-based features from user log data.

    Returns:
        DataFrame: Aggregated merchant-based features.
    """

    # Base merchant-level aggregation
    merchant_features = user_log.groupby('seller_id').agg({
        'user_id': 'nunique',  # Total unique users
        'item_id': 'count',    # Total interactions
        'cat_id': 'nunique',   # Unique categories
        'brand_id': 'nunique', # Unique brands
    }).reset_index()

    # Rename columns for clarity
    merchant_features.rename(columns={
        'seller_id': 'merchant_id',
        'user_id': 'total_users',
        'item_id': 'total_interactions',
        'cat_id': 'unique_categories',
        'brand_id': 'unique_brands'
    }, inplace=True)

    # User engagement metrics
    merchant_user_counts = user_log.groupby('seller_id')['user_id'].value_counts().groupby('seller_id').agg(['mean', 'std', 'max']).reset_index()
    merchant_user_counts.rename(columns={
        'mean': 'avg_interactions_per_user',
        'std': 'std_interactions_per_user',
        'max': 'max_interactions_by_single_user'
    }, inplace=True)

    merchant_features = merchant_features.merge(
        merchant_user_counts[['seller_id', 'avg_interactions_per_user', 'std_interactions_per_user', 'max_interactions_by_single_user']],
        left_on='merchant_id', right_on='seller_id', how='left'
    ).drop('seller_id', axis=1)

    # Popularity metrics
    merchant_features['user_interaction_ratio'] = merchant_features['total_users'] / merchant_features['total_interactions']

    # Diversity metrics
    merchant_features['category_concentration'] = user_log.groupby('seller_id')['cat_id'].apply(
        lambda x: x.value_counts(normalize=True).max()
    ).values

    merchant_features['brand_concentration'] = user_log.groupby('seller_id')['brand_id'].apply(
        lambda x: x.value_counts(normalize=True).max()
    ).values

    # Behavioral metrics
    merchant_features['repeat_user_ratio'] = user_log.groupby('seller_id')['user_id'].apply(
        lambda x: (x.value_counts() > 1).sum() / len(x)
    ).values

    merchant_features['action_entropy'] = user_log.groupby('seller_id')['action_type'].apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True))))
    ).values

    # Conversion metrics (if purchases exist)
    merchant_action_counts = user_log.groupby('seller_id')['action_type'].value_counts().unstack(fill_value=0).reset_index()
    merchant_action_counts.columns = ['merchant_id', 'total_clicks', 'total_carts', 'total_purchases', 'total_favs']

    merchant_features = merchant_features.merge(merchant_action_counts, on='merchant_id', how='left')

    merchant_features['purchase_ratio'] = merchant_features['total_purchases'] / merchant_features['total_interactions']
    merchant_features['cart_conversion_rate'] = merchant_features['total_purchases'] / (merchant_features['total_carts'] + 1e-9)
    merchant_features['favorite_conversion_rate'] = merchant_features['total_purchases'] / (merchant_features['total_favs'] + 1e-9)

    return merchant_features


def prepare_pipeline():
    """
    Prepare the data processing pipeline by merging and engineering features.

    Returns:
        tuple: Processed training dataset, test dataset, and list of filtered feature columns.
    """

    train, user_info, user_log, test = get_data()

    # Merge user info with train dataset
    train = train.merge(user_info, on='user_id', how='left')
    test = test.merge(user_info, on='user_id', how='left')

    user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%m%d', errors='coerce')
    user_log['day_of_week'] = user_log['time_stamp'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
    user_log['month'] = user_log['time_stamp'].dt.month            # Month of the action

    time_features = time_based_feature_eng(user_log)
    user_features = user_based_feature_eng(user_log, time_features)
    merchant_features = merchant_based_feature_eng(user_log)
    # Merge all features with train dataset
    train = train.merge(user_features, on='user_id', how='left')
    train = train.merge(merchant_features, on='merchant_id', how='left')

    test = test.merge(user_features, on='user_id', how='left')
    test = test.merge(merchant_features, on='merchant_id', how='left')

    numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    corr_matrix = train[numeric_columns].corr()

    filtered_columns = corr_matrix[abs(corr_matrix['label']) >= 0.01].index.tolist()
    filtered_columns = [col for col in filtered_columns if col != 'label']

    return train, test, filtered_columns

def prepare_data(train, test, filtered_columns):
    """
    Prepare data for model training and testing by imputing missing values, scaling, and splitting.

    Returns:
        tuple: Scaled training, validation, and test sets along with labels.
    """

    # Prepare data
    X = train[filtered_columns]
    y = train['label']
    X_test = test[filtered_columns]

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y = train['label']

    X_test = imputer.transform(X_test)

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    X_test_scaled = scaler_full.transform(X_test)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val , X_scaled, y, X_test_scaled

def main():
    train, test, filtered_columns = prepare_pipeline()
    X_train, X_val, y_train, y_val , X_scaled, y, X_test_scaled = prepare_data(train, test, filtered_columns)

    # Get Predictions from Ensemble Model
    ensemble_preds_weighted = non_nn_models.weighted_ensemble(X_train, y_train, X_val, y_val, X_scaled, y, X_test_scaled)

    test['prob'] = ensemble_preds_weighted

    # Prepare submission file
    submission = test[['user_id', 'merchant_id', 'prob']]
    submission.to_csv('ensemble_predictions.csv', index=False)
    print("Ensemble Submission file created.")

    # Get predictions from Neural Network
    net = neural_network.train_network(X_train, y_train, X_val, y_val)
    nn_test_predictions = neural_network.predict_test(net, X_test_scaled)

    test['prob'] = nn_test_predictions

    # Prepare submission file
    submission = test[['user_id', 'merchant_id', 'prob']]
    submission.to_csv('nn_predictions.csv', index=False)
    print("Neural Network Submission file created.")



if __name__ == "__main__":
    main()


    


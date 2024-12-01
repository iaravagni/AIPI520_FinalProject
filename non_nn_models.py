from sklearn.metrics import roc_auc_score

# Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


# Optimization
from hyperopt import fmin, tpe, hp, STATUS_OK

def optimize_lr(X_train, y_train, X_val, y_val):
    """
    Optimize Logistic Regression hyperparameters using Hyperopt.

    Parameters:
        X_train (ndarray): Training feature data.
        y_train (ndarray): Training labels.
        X_val (ndarray): Validation feature data.
        y_val (ndarray): Validation labels.

    Returns:
        dict: Best hyperparameters for Logistic Regression.
    """

    space_lr = {
        'C': hp.uniform('C', 0.01, 1),  # Regularization strength
    }

    def objective_lr(params):
        lr = LogisticRegression(**params, random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        return {'loss': -score, 'status': STATUS_OK}

    best_params_lr = fmin(objective_lr, space_lr, algo=tpe.suggest, max_evals=50)

    return best_params_lr


def optimize_dt(X_train, y_train, X_val, y_val):
    """
    Optimizes Decision Tree hyperparameters using Hyperopt.
    """
    space_dt = {
        'max_depth': hp.quniform('max_depth', 1, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
    }

    def objective_dt(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        dt = DecisionTreeClassifier(**params, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        return {'loss': -score, 'status': STATUS_OK}

    best_params_dt = fmin(objective_dt, space_dt, algo=tpe.suggest, max_evals=50)
    print("Best Decision Tree hyperparameters:", best_params_dt)

    for k, v in best_params_dt.items():
        if k == 'max_depth' or k == 'min_child_samples' or k == 'min_child_split' or k=='min_samples_leaf' or k=='min_samples_split':
            best_params_dt[k] = int(v)
        if k == 'criterion':
            best_params_dt[k] = 'gini'

    return best_params_dt


def optimize_xgb(X_train, y_train, X_val, y_val):
    """
    Optimizes XGBoost hyperparameters using Hyperopt.
    """
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 100, 1),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
        'max_depth': hp.quniform('max_depth', 2, 8, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
        'subsample': hp.uniform('subsample', 0.5, 1), 
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), 
        'gamma': hp.uniform('gamma', 0, 5),                        
        'reg_alpha': hp.uniform('reg_alpha', 0, 5),                
        'reg_lambda': hp.uniform('reg_lambda', 0, 5),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 5, 25)        
    }


    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])

        xgb_model = XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        return {'loss': -score, 'status': STATUS_OK}

    best_params_xgb = fmin(objective, space, algo=tpe.suggest, max_evals=100)
    print("Best set of hyperparameters: ", best_params_xgb)

    for k, v in best_params_xgb.items():
        if k == 'max_depth' or k == 'min_child_samples' or k == 'n_estimators':
            best_params_xgb[k] = int(v)

    return best_params_xgb

def optimize_lgb(X_train, y_train, X_val, y_val):
    """
    Optimizes LightGBM hyperparameters using Hyperopt.
    """
    space_lgb = {
            'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'max_depth': hp.quniform('max_depth', 2, 20, 1),
            'num_leaves': hp.quniform('num_leaves', 20, 200, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 5),
            'reg_lambda': hp.uniform('reg_lambda', 0, 5),
        }

    def objective_lgb(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['num_leaves'] = int(params['num_leaves'])
        params['min_child_weight'] = int(params['min_child_weight'])

        lgb = LGBMClassifier(**params, random_state=42)
        lgb.fit(X_train, y_train)
        y_pred = lgb.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        return {'loss': -score, 'status': STATUS_OK}

    best_params_lgb = fmin(objective_lgb, space_lgb, algo=tpe.suggest, max_evals=50)
    print("Best LightGBM hyperparameters:", best_params_lgb)

    for k, v in best_params_lgb.items():
        if k == 'max_depth' or k == 'min_child_samples' or k == 'n_estimators' or k == 'num_leaves':
            best_params_lgb[k] = int(v)

    return best_params_lgb



def train_models(X_train, y_train, X_val, y_val, X_scaled, y):
    """
    Train multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
    using optimized hyperparameters.
    """
    best_params_lr = optimize_lr(X_train, y_train, X_val, y_val)
    best_params_dt = optimize_dt(X_train, y_train, X_val, y_val)
    best_params_rf = best_params_dt
    best_params_xgb = optimize_xgb(X_train, y_train, X_val, y_val)
    best_params_lgb = optimize_lgb(X_train, y_train, X_val, y_val)

    lr_model = LogisticRegression(**best_params_lr)
    dt_model = DecisionTreeClassifier(**best_params_dt)
    rf_model = RandomForestClassifier(**best_params_rf)
    xgb_model = XGBClassifier(**best_params_xgb)
    lgb_model = LGBMClassifier(**best_params_lgb)

    # Train all models
    lr_model.fit(X_scaled, y)
    dt_model.fit(X_scaled, y)
    rf_model.fit(X_scaled, y)
    xgb_model.fit(X_scaled, y)
    lgb_model.fit(X_scaled, y)

    return lr_model, dt_model, rf_model, xgb_model, lgb_model

def weighted_ensemble(X_train, y_train, X_val, y_val, X_scaled, y, X_test):
    """
    Creates a weighted ensemble of predictions from multiple models.

    Parameters:
        X_train (ndarray): Training feature data.
        y_train (ndarray): Training labels.
        X_val (ndarray): Validation feature data.
        y_val (ndarray): Validation labels.
        X_scaled (ndarray): Entire feature data (scaled).
        y (ndarray): Entire label data.
        X_test (ndarray): Test feature data.

    Returns:
        ndarray: Weighted ensemble predictions for the test data.
    """

    lr_model, dt_model, rf_model, xgb_model, lgb_model = train_models(X_train, y_train, X_val, y_val, X_scaled, y)

    # Get predictions (as probabilities)
    lr_preds = lr_model.predict_proba(X_test)[:, 1]
    dt_preds = dt_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]

    # Define weights based on model performance
    weights = [0.1, 0.1, 0.1, 0.4, 0.3]

    # Weighted predictions
    ensemble_preds_weighted = (
        weights[0] * lr_preds +
        weights[1] * dt_preds +
        weights[2] * rf_preds +
        weights[3] * xgb_preds +
        weights[4] * lgb_preds
    )

    return ensemble_preds_weighted

import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """Load data from CSV files."""
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    """Train RandomForestRegressor model and evaluate."""
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Train Data Score:", train_score)
    print("Test Data Score:", test_score)

    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)

    model_name = f"random_forest_model_{np.random.randint(1000)}.joblib"
    pickle.dump(model, open(f"models/{model_name}.pkl", "wb"))
    print("Trained model saved as:", model_name)

    return model, mae, rmse, r2


def tune_model(model, X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    # Define hyperparameters grid
    param_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
        "max_features": ["auto", "sqrt"],
        "max_depth": [int(x) for x in np.linspace(5, 30, num=6)],
        "min_samples_split": [2, 5, 10, 15, 100],
        "min_samples_leaf": [1, 2, 5, 10],
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="neg_mean_squared_error",
        n_iter=10,
        cv=5,
        verbose=2,
        n_jobs=1,
    )

    random_search.fit(X_train, y_train)

    # Evaluate best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    y_pred = best_model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)

    model_name = f"random_forest_tuned_model_{np.random.randint(1000)}.joblib"
    pickle.dump(best_model, open(f"models/{model_name}.pkl", "wb"))
    print("Tuned model saved as:", model_name)

    return best_model, best_params, mae, rmse, r2


# Example usage:
# X_train, X_test, y_train, y_test = load_data(X_train_path, X_test_path, y_train_path, y_test_path)
# model, mae, rmse, r2 = train_model(X_train, X_test, y_train, y_test)
# tuned_model, best_params, tuned_mae, tuned_rmse, tuned_r2 = tune_model(model, X_train, y_train, X_test, y_test)

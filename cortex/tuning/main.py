from sklearn.model_selection import GridSearchCV
import warnings

def run_hyperparameter_tuning(model_instance, X, y, cv=5):
    """
    Performs hyperparameter tuning using GridSearchCV.
    
    Args:
        model_instance: An instance of the model to tune.
        X: Features.
        y: Target variable.
        cv: Number of cross-validation folds.
        
    Returns:
        A tuple containing the best model instance and the best score.
    """
    if not model_instance.param_grid:
        print(f"No hyperparameter grid defined for {model_instance.name}. Skipping tuning.")
        return model_instance.model, None

    print("\nStarting hyperparameter tuning with GridSearchCV...")
    print(f"Searching over the following parameters: {model_instance.param_grid}")

    # For now, we'll use a simple accuracy metric. We'll make this user-configurable later.
    scoring_metric = 'accuracy' if hasattr(model_instance.model, 'predict_proba') else 'neg_mean_squared_error'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore warnings from GridSearch
        grid_search = GridSearchCV(
            estimator=model_instance.model,
            param_grid=model_instance.param_grid,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1, # Use all available CPU cores
            verbose=1
        )
        
        grid_search.fit(X, y)
        
    print("\nHyperparameter tuning complete!")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_score_
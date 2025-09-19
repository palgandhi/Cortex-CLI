import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
import torch
from fuzzywuzzy import fuzz
from cortex.nlp.parser import PROBLEM_TYPE_KEYWORDS
from cortex.tuning.main import run_hyperparameter_tuning
from cortex.algorithms.base import BaseModel
from cortex.algorithms.deep_learning.base import BaseDeepLearningModel
from torch.utils.data import TensorDataset, DataLoader

def get_target_column(data, problem_type, user_input):
    """
    Intelligently suggests or prompts for the target column.
    """
    inferred_column = None
    if problem_type in PROBLEM_TYPE_KEYWORDS:
        keywords = PROBLEM_TYPE_KEYWORDS[problem_type]
        for column in data.columns:
            if any(fuzz.partial_ratio(column.lower(), keyword) > 90 for keyword in keywords):
                inferred_column = column
                break

    if not inferred_column and 'target' in data.columns:
        inferred_column = 'target'

    if inferred_column:
        confirmation = input(
            f"I found a column named '{inferred_column}'. Is this the target variable? (yes/no): "
        ).lower().strip()
        if confirmation in ('yes', 'y'):
            return inferred_column
        else:
            print("Okay, please provide the name of the target column.")

    print("\nI couldn't infer the target column from your input.")
    print("Available columns are:")
    print(list(data.columns))

    while True:
        target_column_name = input("Please enter the name of the target column: ").strip()
        if target_column_name in data.columns:
            return target_column_name
        else:
            print(f"Error: The column '{target_column_name}' was not found. Please try again.")


def run_training_pipeline(handler, model_class, problem_type, user_input, auto_run=False):
    """
    Runs the full ML pipeline and returns the evaluation metrics.
    """
    try:
        handler.load_data()
        data = handler.data

        # --- Handle RL Separately ---
        if problem_type == "reinforcement_learning":
            env = handler.data
            print(f"Using the '{handler.env_id}' environment for reinforcement learning.")
            final_model_instance = model_class()
            print(f"Training '{final_model_instance.name}'...")
            final_model_instance.train(env)
            print("Model training complete. Evaluating...")
            metrics = final_model_instance.evaluate(env)
            
            # --- Return metrics for comparison ---
            if auto_run:
                return metrics
            
            print("\n--- Evaluation Results ---")
            # ... (existing print logic)
            save_choice = input("\nWould you like to save the trained model? (yes/no): ").lower().strip()
            # ... (existing save logic)
            return metrics
        
        # --- Handle Clustering ---
        elif problem_type == "clustering":
            if not isinstance(data, pd.DataFrame) or len(data) < 10:
                print("\nWarning: The dataset is too small for a meaningful evaluation.")
                return None
            
            print(f"Using the entire dataset for unsupervised learning.")
            X = data.copy()
            y = None

            final_model_instance = model_class()
            print(f"Training '{final_model_instance.name}'...")
            final_model_instance.train(X)
            print("Model training complete. Evaluating...")
            metrics = final_model_instance.evaluate(X)
            
            if auto_run:
                return metrics
            
            print("\n--- Evaluation Results ---")
            # ... (existing print logic)
            save_choice = input("\nWould you like to save the trained model? (yes/no): ").lower().strip()
            # ... (existing save logic)
            return metrics

        # --- Handle All Supervised Learning ---
        else:
            if not isinstance(data, pd.DataFrame) or len(data) < 10:
                print("\nWarning: The dataset is too small for a meaningful train-test split.")
                print("Please use a larger dataset for more reliable model evaluation.")
                return None

            if handler.detect_type() == "text":
                target_column = get_target_column(data, problem_type, user_input) if not auto_run else data.columns[-1]
                if not target_column: return None
                X, y = handler.get_features_and_target(target_column)
                print("Casting target column to categorical data type...")
                y = y.astype('category')
            else: # Tabular data
                target_column = get_target_column(data, problem_type, user_input) if not auto_run else data.columns[-1]
                if not target_column: return None
                
                if problem_type == "classification":
                    print("Casting target column to categorical data type...")
                    data[target_column] = data[target_column].astype('category')
                    num_classes = len(data[target_column].cat.categories)
                    if num_classes > 2:
                        print(f"Detected {num_classes} classes. Proceeding with multi-class classification.")
                    else:
                        print("Detected binary classification.")
                elif problem_type == "regression":
                    print("Casting target column to numeric data type...")
                    data[target_column] = pd.to_numeric(data[target_column])
                
                print(f"Using '{target_column}' as the target variable.")
                X = data.drop(columns=[target_column])
                y = data[target_column]

            is_deep_learning = issubclass(model_class, BaseDeepLearningModel)
            final_model_instance = model_class()

            if is_deep_learning:
                print("Hyperparameter tuning for deep learning models is not yet implemented.")

                X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32).view(-1, 1)
                X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test_df.values, dtype=torch.float32).view(-1, 1)

                train_dataloader = DataLoader(
                    TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
                )
                test_dataloader = DataLoader(
                    TensorDataset(X_test_tensor, y_test_tensor), batch_size=32
                )

                print(f"Training '{final_model_instance.name}'...")
                final_model_instance.train(train_dataloader)
                print("Model training complete. Evaluating...")
                metrics = final_model_instance.evaluate(test_dataloader)

            else:
                if final_model_instance.param_grid:
                    print("Hyperparameter tuning is enabled. Automatically running tuning...")
                    best_model, _ = run_hyperparameter_tuning(final_model_instance, X, y)
                    final_model_instance.model = best_model
                    print("Using best model found during tuning for final training and evaluation.")
                else:
                    print("Skipping hyperparameter tuning. Training with default parameters.")

                print("\nSplitting data into training and testing sets...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                print(f"Training '{final_model_instance.name}'...")
                final_model_instance.train(X_train, y_train)
                print("Model training complete. Evaluating...")
                metrics = final_model_instance.evaluate(X_test, y_test)
            
            if auto_run:
                return metrics

            print("\n--- Evaluation Results ---")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
            print("--------------------------")

            save_choice = input("\nWould you like to save the trained model? (yes/no): ").lower().strip()
            if save_choice in ('yes', 'y'):
                save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'cortex_model_v1.pkl')
                final_model_instance.save(save_path)
                print(f"Model saved to {save_path}")
            else:
                print("Model not saved.")
            
            return metrics

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
Cortex CLI
The Command-Line ML Engine

!

Cortex CLI is a powerful and intuitive command-line tool designed to help developers and students quickly prototype and train machine learning models. It automates the entire ML pipeline from the terminal, making it ideal for hackathons and rapid experimentation.

Features
Natural Language Processing (NLP): Understands user intent and problem type using a dynamic, lightweight NLP model, making the CLI intelligent and easy to use.

Automated Pipeline: The CLI handles best practices like hyperparameter tuning and cross-validation automatically, so you don't have to.

Extensive Model Library: A modular architecture allows for the easy integration of 50+ models, including advanced algorithms from scikit-learn, XGBoost, and LightGBM.

Multi-Task Support: Handles supervised (regression, classification), unsupervised (clustering), and reinforcement learning tasks with a single, unified interface.

Model Comparison: The all command allows you to automatically train and evaluate multiple models and compare their performance.

Flexible Execution: Supports both interactive mode for guidance and a "one-shot" mode for fast, scripted execution.

Installation
To get started with Cortex, clone the repository and install it using pip. This will make the cortex command available globally on your system.

1. Clone the repository:

Bash
git clone https://github.com/your-username/cortex.git
cd cortex
2. Install the package:

Bash
pip install .
This command will install Cortex and all its dependencies, including scikit-learn, xgboost, lightgbm, gymnasium, and sentence-transformers.

Usage
Simply type cortex in your terminal to start the interactive session.

Interactive Mode
Follow the prompts to guide the CLI through your machine learning task.

Bash
cortex

# Please enter the path to your dataset: sample_data.csv
# What do you want to do with this dataset?: I want to predict the salary
# ...
One-Shot Mode
For a fast, non-interactive experience, use the --auto-run flag. This will run the pipeline with the first suggested model and default settings.

Bash
cortex --auto-run path/to/your/dataset.csv
Model Comparison
In interactive mode, you can run all suggested models at once by typing all when prompted.

Bash
Please select a model by number (1-3) or type 'all' to run all: all
Models Included
Cortex comes pre-configured with a powerful set of models, organized by problem type:

Regression

XGBoost Regressor: A powerful gradient boosting model for continuous values.

LightGBM Regressor: A fast and efficient gradient boosting model.

Linear Regression: A simple, fast baseline model.

Classification

XGBoost Classifier: A top choice in hackathons for its speed and accuracy.

LightGBM Classifier: An efficient classifier ideal for large datasets.

Random Forest Classifier: A robust ensemble model for non-linear data.

Ensemble Voting Classifier: Combines multiple models for improved accuracy.

Reinforcement Learning

Q-Learning Agent: A classic algorithm for tabular environments.

Clustering

K-Means Clustering: A popular algorithm for finding groups in data.

Contributing
We welcome contributions! If you have an idea for a new feature, a bug fix, or a new model to add, please feel free to open a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
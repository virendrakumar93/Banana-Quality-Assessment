# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

# Function Definitions
def preprocess_data(file_path):
    """Preprocess the dataset and return X, y, and column details."""
    df = pd.read_csv(file_path)
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['harvest_month'] = df['harvest_date'].dt.month
    num_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['sample_id']).columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).drop(columns=['harvest_date']).columns
    df = df.drop(columns=['sample_id', 'harvest_date'])
    
    label_encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    X = df.drop('quality_category', axis=1)
    y = df['quality_category']
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train models, evaluate them, and return results."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis": LDA(),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Bagging Classifier": BaggingClassifier(random_state=42)
    }

    param_grids = {
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
        "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
        "CatBoost": {"iterations": [100, 200], "learning_rate": [0.01, 0.1], "depth": [3, 5]},
        "Decision Tree": {"max_depth": [5, 10, None]},
    }

    results = []

    for name, model in models.items():
        print(f"Running {name}...")
        if name in param_grids:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grids[name],
                cv=skf, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best Parameters for {name}: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Evaluate on test data
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Test Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Print Metrics
        print(f"\n{name} Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    return results

def main():
    # Load and preprocess data
    file_path = '/kaggle/input/banana-quality-dataset/banana_quality_dataset.csv'
    X, y = preprocess_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train_smote, X_test, y_train_smote, y_test)

    # Voting Classifier
    print("\nRunning Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ("Random Forest", RandomForestClassifier(random_state=42)),
            ("AdaBoost", AdaBoostClassifier(random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
            ("CatBoost", CatBoostClassifier(random_state=42, verbose=0))
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_smote, y_train_smote)
    y_pred_voting = voting_clf.predict(X_test)

    # Evaluate Voting Classifier
    accuracy = accuracy_score(y_test, y_pred_voting)
    precision = precision_score(y_test, y_pred_voting, average='weighted')
    recall = recall_score(y_test, y_pred_voting, average='weighted')
    f1 = f1_score(y_test, y_pred_voting, average='weighted')
    cm_voting = confusion_matrix(y_test, y_pred_voting)

    print("\nVoting Classifier Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_voting))

    # Plot Confusion Matrix for Voting Classifier
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_voting, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Voting Classifier Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Display Results Summary
    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df)

    # Visualize model performances
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Test Accuracy", data=results_df, palette="viridis")
    plt.title("Model Comparison - Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()

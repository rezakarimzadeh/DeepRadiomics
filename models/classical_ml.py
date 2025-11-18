import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier

def train_classical_ml_models(data_root, fold_index):
    # ---------------------------------------------------------------------
    # radiomics_features: [N_samples, N_features]
    # labels: array-like of shape [N_samples], values in {0,1}
    # ---------------------------------------------------------------------

    # Define models in pipelines with scaling where needed
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF kernel)": make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "k-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
        ),
        "lightGBM":  LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=500, random_state=42),
        # "TabNet": TabNetClassifier(),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=42)
        )
    }


    # Train & evaluate each model
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(name, y_test, y_pred, y_prob)
        results.append(metrics)
        # print(f"\n=== {name} ===")
        # print(classification_report(y_test, y_pred, digits=3))
    #     print(f"ROC-AUC: {auc:.3f}")
    #     RocCurveDisplay.from_predictions(y_test, y_prob, name=name)

    # plt.show()

    # Summary table
    df_results = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\nModel Performance Summary:")
    print(df_results)
    return df_results

def mlp_fivefold_cc():
    masih_list = []
    razavi_list = []
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        config.fold_number = fold_idx
        masih_results, razavi_results = train_model()
        masih_list.append(masih_results)
        razavi_list.append(razavi_results)
    # Aggregate results
    masih_aggregated = pd.DataFrame(masih_list).groupby("model").agg(
        Accuracy_Mean=("accuracy", "mean"),
        Accuracy_STD=("accuracy", "std"),
        ROC_AUC_Mean=("roc_auc", "mean"),
        ROC_AUC_STD=("roc_auc", "std"),
        Precision_Mean=("precision", "mean"),
        Precision_STD=("precision", "std"),
        Recall_Mean=("recall", "mean"),
        Recall_STD=("recall", "std"),
        Specificity_Mean=("specificity", "mean"),
        Specificity_STD=("specificity", "std"),
        F1_Score_Mean=("f1_score", "mean"),
        F1_Score_STD=("f1_score", "std"),
    ).reset_index()
    print("\nAggregated MLP Model Performance on Masih Dataset over 5 folds:")
    print(masih_aggregated)

def fivefold_classical_ml():
    results_list = []
    for fold_idx in range(5):
        print(f"Training classical ML models for fold {fold_idx+1}/5")
        fold_results = train_classical_ml_models(config.data_root, fold_index=fold_idx)
        results_list.append(fold_results)
    # Aggregate results
    aggregated_results = pd.concat(results_list).groupby("model").agg(
        Accuracy_Mean=("accuracy", "mean"),
        Accuracy_STD=("accuracy", "std"),
        ROC_AUC_Mean=("roc_auc", "mean"),
        ROC_AUC_STD=("roc_auc", "std"),
        Precision_Mean=("precision", "mean"),
        Precision_STD=("precision", "std"),
        Recall_Mean=("recall", "mean"),
        Recall_STD=("recall", "std"),
        Specificity_Mean=("specificity", "mean"),
        Specificity_STD=("specificity", "std"),
        F1_Score_Mean=("f1_score", "mean"),
        F1_Score_STD=("f1_score", "std"),
    ).reset_index()
    aggregated_results = aggregated_results.round(4)
    print("\nAggregated Classical ML Model Performance over 5 folds:")
    print(aggregated_results)
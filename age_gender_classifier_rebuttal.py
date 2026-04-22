
import argparse
import shutil
import pytorch_lightning as pl
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from models.classical_ml import get_ml_models, get_param_grids
from dataloaders.ml_dataloaders import get_demographic_ml_center1, get_demographic_ml_center2
from utils import read_yaml_file, test_model, compute_classification_metrics, save_json, test_model_graph
from pathlib import Path
import os
import pandas as pd
import shutil
import argparse
pl.seed_everything(42)

def train_ml_model(argparse, fold_index: int):
    X_train, y_train, X_test, y_test = get_demographic_ml_center1(argparse, fold_index)
    center2_X_test, center2_y_test = get_demographic_ml_center2(argparse)
    models = get_ml_models()
    
    if argparse.use_age and argparse.use_gender:
        print(f"Training classical ML models with coords, age and gender, Fold {fold_index}")
        save_root = Path("Results") / "Rebuttal" / "ml_models_age_gender"
    elif argparse.use_age:
        print(f"Training classical ML models with age, Fold {fold_index}")
        save_root = Path("Results") / "Rebuttal" / "ml_models_age"
    elif argparse.use_gender:
        print(f"Training classical ML models with gender, Fold {fold_index}")
        save_root = Path("Results") / "Rebuttal" / "ml_models_gender"
    else:
        raise ValueError("At least one of --use_age or --use_gender must be True.")

    save_dir = save_root / f"fold_{fold_index}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    fold_results = {}

    center2_results = []
    center2_fold_results = {}
    
    # for name, model in models.items():
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # y_prob = model.predict_proba(X_test)[:, 1]

    # grid search for hyperparameter tuning
    param_grids = get_param_grids()
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Tuning and training: {name}")

        param_grid = param_grids.get(name, None)

        if param_grid is not None:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_inner_score = search.best_score_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
            best_inner_score = None

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = compute_classification_metrics(name, y_test, y_pred, y_prob)
        results.append(metrics)

        fold_results[name] = {'fold': fold_index, **metrics, 
                              'model_name': name,
                              'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
        save_json(save_dir / f"results_fold_{fold_index}.json", fold_results)

        # Center2 test
        center2_y_pred = best_model.predict(center2_X_test)
        center2_y_prob = best_model.predict_proba(center2_X_test)[:, 1]
        center2_metrics = compute_classification_metrics(name, center2_y_test, center2_y_pred, center2_y_prob)
        center2_results.append(center2_metrics)
        center2_fold_results[name] = {'info': "Center2 Test Results without TTA",
                                      'fold': fold_index, **center2_metrics, 
                                       'model_name': name,
                                      'y_true': center2_y_test.tolist(), 'y_pred': center2_y_pred.tolist(), 'y_prob': center2_y_prob.tolist()}
        save_json(save_dir / f"center2_results_fold_{fold_index}.json", center2_fold_results)

    df_results = pd.DataFrame(results)
    center2_df_results = pd.DataFrame(center2_results)
    print(f"\nModel Performance Summary fold {fold_index}:")
    print(df_results)
    return df_results, center2_df_results, save_root

def fivefold_cv(argparse):
    aggrigation_list = []
    center2_aggrigation_list = []
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        if argparse.model_name == "ml_models":
            classification_results, center2_classification_results, model_save_path = train_ml_model(argparse, fold_idx)
        else:
            raise ValueError(f"Model {argparse.model_name} not recognized.")
        aggrigation_list.append(classification_results)
        center2_aggrigation_list.append(center2_classification_results)

    aggregated_metrics = pd.concat(aggrigation_list).groupby("model").agg(
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
    print("\nAggregated Model Performance on Masih Dataset over 5 folds:")
    # aggregated_metrics['use_coords'] = argparse.use_coords
    print(aggregated_metrics)
    save_json(model_save_path / "aggregated_results.json", aggregated_metrics.to_dict(orient="records"))

    center2_aggregated_metrics = pd.concat(center2_aggrigation_list).groupby("model").agg(
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
    print("\nAggregated Model Performance on Center2 Dataset over 5 folds:")
    # center2_aggregated_metrics['use_coords'] = argparse.use_coords
    print(center2_aggregated_metrics)
    save_json(model_save_path / "center2_aggregated_results.json", center2_aggregated_metrics.to_dict(orient="records"))


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("True", "yes", "true", "t", "1"):
            return True
        elif v.lower() in ("False""no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_root", type=str, default="/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset", help="Base directory for the dataset.")
    parser.add_argument("--model_name", type=str, default="ml_models", choices=["ml_models"], help="Name of the model to train.")
    parser.add_argument("--use_age", type=str2bool, default=True, help="Whether to use age features.")
    parser.add_argument("--use_gender", type=str2bool, default=True, help="Whether to use gender features.")
    args = parser.parse_args()
    fivefold_cv(args)

if __name__ == "__main__":
    main()
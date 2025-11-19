import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.deep_sets import RadiomicsDeepSets
from models.transformer import RadiomicsTransformer
from models.MIL import RadiomicsMIL
from models.graph import RadiomicsGraph
from models.classical_ml import get_ml_models
from dataloaders.deep_dataloaders import get_dataloaders_deep_learning
from dataloaders.graph_dataloader import get_dataloaders_graph
from dataloaders.ml_dataloaders import get_dataloaders_ml
from utils import read_yaml_file, test_model, compute_classification_metrics, save_json, test_model_graph
from pathlib import Path
import os
import pandas as pd
import shutil
import argparse
pl.seed_everything(42)

def model_generator(model_name: str):
    if model_name == "transformer":
        return RadiomicsTransformer
    elif model_name == "deep_sets":
        return RadiomicsDeepSets
    elif model_name == "mil":
        return RadiomicsMIL
    elif model_name == "graph":
        return RadiomicsGraph
    else:
        raise ValueError(f"Model {model_name} not found in model zoo.")

def train_dl_model(argparse, fold_index: int):
    config_base_dir = './configs'
    model_configs = read_yaml_file(Path(config_base_dir) / f"{argparse.model_name}.yaml")
    if argparse.model_name == "graph":
        train_loader, val_loader, test_loader = get_dataloaders_graph(argparse.data_root, argparse.use_tda, batch_size=model_configs['batch_size'], fold_index=fold_index)
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.num_node_features
    else:
        train_loader, val_loader, test_loader = get_dataloaders_deep_learning(argparse.data_root, argparse.use_tda, batch_size=model_configs['batch_size'], fold_index=fold_index)
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[-1]
    MODEL = model_generator(argparse.model_name)
    model = MODEL(input_dim=input_dim, config=model_configs)
    print("================= Training Configuration ================")
    print(f"Input feature dimension: {input_dim}, Model: {argparse.model_name}, Fold: {fold_index}, TDA: {argparse.use_tda}")
    

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,       
        filename="best",     
        auto_insert_metric_name=False,
        # verbose=True,
    )
    if argparse.use_tda:
        log_name = f"{argparse.model_name}_tda_radiomics"
    else:
        log_name = f"{argparse.model_name}_radiomics"

    save_dir = os.path.join("Results", log_name, f"fold_{fold_index}")
    if os.path.exists(save_dir):
        # delete existing directory
        shutil.rmtree(save_dir)
    #  TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="Results", name=log_name, version=f"fold_{fold_index}")

    #  Trainer (300 epochs)
    trainer = pl.Trainer(
            max_epochs=model_configs['max_epochs'],
            callbacks=[ckpt],
            logger=tb_logger,
            accelerator="auto",
            devices="auto",
            )
    #  Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}")
    #  Test
    best_model = MODEL.load_from_checkpoint(ckpt.best_model_path, config=model_configs, input_dim=input_dim)
    if argparse.model_name == "graph":
        y_true, y_pred, y_prob = test_model_graph(best_model, test_loader)
    else:
        y_true, y_pred, y_prob = test_model(best_model, test_loader)
    classification_results = compute_classification_metrics(argparse.model_name, y_true, y_pred, y_prob)
    print(f"Test Results Fold {fold_index}: {classification_results}")
    fold_results = {'fold': fold_index, **classification_results, 'best_checkpoint': ckpt.best_model_path,
                    'use_tda': argparse.use_tda, 'model_name': argparse.model_name, 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)
    return pd.DataFrame([fold_results]), Path("Results") / log_name 


def train_ml_model(argparse, fold_index: int):
    X_train, y_train, X_test, y_test = get_dataloaders_ml(argparse, fold_index)
    models = get_ml_models()
    
    if argparse.use_tda:
        print(f"Training classical ML models with TDA features, Fold {fold_index}")
        save_root = Path("Results") / "ml_models_tda_radiomics"
    else:
        print(f"Training classical ML models with Radiomics features, Fold {fold_index}")
        save_root = Path("Results") / "ml_models_radiomics"
    
    save_dir = save_root / f"fold_{fold_index}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    fold_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(name, y_test, y_pred, y_prob)
        results.append(metrics)

        fold_results[name] = {'fold': fold_index, **metrics, 
                              'use_tda': argparse.use_tda, 'model_name': name,
                              'y_true': y_test.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
        save_json(save_dir / f"results_fold_{fold_index}.json", fold_results)
    df_results = pd.DataFrame(results)
    print(f"\nModel Performance Summary fold {fold_index}:")
    print(df_results)
    return df_results, save_root

def fivefold_cv(argparse):
    aggrigation_list = []
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        if argparse.model_name in ["transformer", "deep_sets", "mil", "graph"]:
            classification_results, model_save_path = train_dl_model(argparse, fold_idx)
        elif argparse.model_name == "ml_models":
            classification_results, model_save_path = train_ml_model(argparse, fold_idx)
        else:
            raise ValueError(f"Model {argparse.model_name} not recognized.")
        aggrigation_list.append(classification_results)
    # Aggregate results
    # aggregated_metrics = pd.DataFrame(aggrigation_list).groupby("model").agg(
    #     Accuracy_Mean=("accuracy", "mean"),
    #     Accuracy_STD=("accuracy", "std"),
    #     ROC_AUC_Mean=("roc_auc", "mean"),
    #     ROC_AUC_STD=("roc_auc", "std"),
    #     Precision_Mean=("precision", "mean"),
    #     Precision_STD=("precision", "std"),
    #     Recall_Mean=("recall", "mean"),
    #     Recall_STD=("recall", "std"),
    #     Specificity_Mean=("specificity", "mean"),
    #     Specificity_STD=("specificity", "std"),
    #     F1_Score_Mean=("f1_score", "mean"),
    #     F1_Score_STD=("f1_score", "std"),
    # ).reset_index()
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
    print("\nAggregated MLP Model Performance on Masih Dataset over 5 folds:")
    # aggregated_metrics['use_tda'] = argparse.use_tda
    print(aggregated_metrics)
    save_json(model_save_path / "aggregated_results.json", aggregated_metrics.to_dict(orient="records"))

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
    parser.add_argument("--model_name", type=str, default="mil", choices=["transformer", "deep_sets", "mil", "ml_models", 'graph'], help="Name of the model to train.")
    parser.add_argument("--use_tda", type=str2bool, default=False, help="Whether to use TDA features.")
    args = parser.parse_args()
    print(args)
    fivefold_cv(args)

if __name__ == "__main__":
    main()

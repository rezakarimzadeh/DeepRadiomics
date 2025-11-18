import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.deep_sets import RadiomicsDeepSets
from models.transformer import RadiomicsTransformer
from models.MIL import RadiomicsMIL
# from models.classical_ml import ClassicalMLModel
from dataloaders.deep_dataloaders import get_dataloaders_deep_learning
from dataloaders.ml_dataloaders import get_dataloaders_ml
from utils import read_yaml_file, test_model, compute_classification_metrics, save_json
from pathlib import Path
import os
import pandas as pd
import shutil
import argparse

def model_generator(model_name: str):
    if model_name == "transformer":
        return RadiomicsTransformer
    elif model_name == "deep_sets":
        return RadiomicsDeepSets
    elif model_name == "mil":
        return RadiomicsMIL
    else:
        raise ValueError(f"Model {model_name} not found in model zoo.")

def train_model(argparse, fold_index: int):
    config_base_dir = './configs'
    model_configs = read_yaml_file(Path(config_base_dir) / f"{argparse.model_name}.yaml")
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

    save_dir = os.path.join("tb_logs", log_name, f"fold_{fold_index}")
    if os.path.exists(save_dir):
        # delete existing directory
        shutil.rmtree(save_dir)
    #  TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=log_name, version=f"fold_{fold_index}")

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
    y_true, y_pred, y_prob = test_model(best_model, test_loader)
    classification_results = compute_classification_metrics(argparse.model_name, y_true, y_pred, y_prob)
    print(f"Test Results Fold {fold_index}: {classification_results}")
    fold_results = {'fold': fold_index, **classification_results, 'best_checkpoint': ckpt.best_model_path,
                    'use_tda': argparse.use_tda, 'model_name': argparse.model_name, 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
    save_json(Path(save_dir) / f"results_fold_{fold_index}.json", fold_results)
    return classification_results, Path("tb_logs") / log_name 

def fivefold_cv(argparse):
    aggrigation_list = []
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        classification_results, model_save_path = train_model(argparse, fold_idx)
        aggrigation_list.append(classification_results)
    # Aggregate results
    masih_aggregated = pd.DataFrame(aggrigation_list).groupby("model").agg(
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
    save_json(model_save_path / "aggregated_results.json", masih_aggregated.to_dict(orient="records"))

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models with 5-fold cross-validation.")
    parser.add_argument("--data_root", type=str, default="/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset", help="Base directory for the dataset.")
    parser.add_argument("--model_name", type=str, default="mil", choices=["transformer", "deep_sets", "mil"], help="Name of the model to train.")
    parser.add_argument("--use_tda", type=bool, default=False, help="Whether to use TDA features.")
    args = parser.parse_args()
    print(args)
    fivefold_cv(args)

if __name__ == "__main__":
    main()

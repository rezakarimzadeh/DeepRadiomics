import torch
from TTA import tent, norm
from dataloaders.deep_dataloaders import get_dataloaders_deep_learning, get_center2_as_test_loader
from dataloaders.graph_dataloader import get_dataloaders_graph, get_center2_as_test_loader_graph
from dataloaders.ml_dataloaders import get_dataloaders_ml, get_classical_test_loader_center2
from models.MIL import RadiomicsMIL
from models.transformer import RadiomicsTransformer
from utils import read_yaml_file, test_model, compute_classification_metrics, save_json, test_model_graph
from pathlib import Path
import pandas as pd
from main import model_generator, data_function_generator_dl

def setup_tent(model, tta_lr, steps=1):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    # print number of trainable parameters out of total parameters
    print(f"Number of trainable parameters for tent: {sum(p.numel() for p in params)} out of {sum(p.numel() for p in model.parameters())}")
    optimizer = setup_optimizer(params, tta_lr)
    tent_model = tent.Tent(model, optimizer,
                           steps=steps,
                           episodic=False)
    return tent_model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    stats, stat_names = norm.collect_stats(model)
    return norm_model

def setup_optimizer(params, tta_lr):
    return torch.optim.Adam(params, lr=tta_lr)

def get_model_dir(cfg):
    if cfg.use_demographic and cfg.use_coords:
        folder_name = f"{cfg.model_name}_coords_demographic_radiomics"
    elif cfg.use_demographic and not cfg.use_coords:
        folder_name = f"{cfg.model_name}_demographic_radiomics"
    elif not cfg.use_demographic and cfg.use_coords:
        folder_name = f"{cfg.model_name}_coords_radiomics"
    else:
        folder_name = f"{cfg.model_name}_radiomics"
    return Path(cfg.results_root) / folder_name 

def evaluate(cfg, fold_index=0):
    config_base_dir = './configs'
    model_root = get_model_dir(cfg)
    ckpt_base_dir = model_root / f"fold_{fold_index}"
    model_configs = read_yaml_file(Path(config_base_dir) / f"{cfg.model_name}.yaml") 
    _, center2_loader_func, test_model_func = data_function_generator_dl(cfg)

    center2_loader = center2_loader_func(cfg)
    
    sample_batch = next(iter(center2_loader))
    if cfg.model_name == "graph":
        input_dim = sample_batch.num_node_features
    else:
        input_dim = sample_batch['features'].shape[-1]

    model  = model_generator(cfg.model_name).load_from_checkpoint(
        checkpoint_path=Path(ckpt_base_dir) / 'checkpoints' / 'best.ckpt',
        input_dim=input_dim,
        config=model_configs)
    if cfg.tta_method == 'norm':
        print("Starting Test Time Adaptation with NORM...")
        tta_model = setup_norm(model)
    else:
        print("Starting Test Time Adaptation with TENT...")
        tta_model = setup_tent(model, model_configs['tta_lr'])
    y_true_center2, y_pred_center2, y_prob_center2 = test_model_func(tta_model, center2_loader)

    center2_classification_results = compute_classification_metrics(cfg.model_name, y_true_center2, y_pred_center2, y_prob_center2)


    center2_fold_results = {'info': "Center2 Test Results without TTA",
                    'fold': 0, **center2_classification_results, "use_demographic": cfg.use_demographic,
                    'use_coords': cfg.use_coords, 'model_name': cfg.model_name, 'y_true': y_true_center2.tolist(), 'y_pred': y_pred_center2.tolist(), 'y_prob': y_prob_center2.tolist()}
    save_json(Path(ckpt_base_dir) / f"tta_{cfg.tta_method}_center2_results_fold_{fold_index}.json", center2_fold_results)
    # print(pd.DataFrame([center2_fold_results]))
    return pd.DataFrame([center2_fold_results]), model_root

def fivefold_cv_tta(args):
    center2_aggrigation_list = []
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        if args.model_name in ["transformer", "deep_sets", "mil", "graph", 'set_transformer']:
            center2_classification_results, model_save_path = evaluate(args, fold_idx)
       
        else:
            raise ValueError(f"Model {args.model_name} not recognized.")
        center2_aggrigation_list.append(center2_classification_results)

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
    save_json(model_save_path / f"tta_{args.tta_method}_center2_aggregated_results.json", center2_aggregated_metrics.to_dict(orient="records"))

if __name__ == '__main__':
    class Args:
        model_name = 'transformer'
        data_root = "/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset"
        results_root = "./Results"
        use_coords = True
        use_demographic = False
        tta_method = 'norm'
    
    cfg = Args()
    for method in ['tent', 'norm']: #, 'norm'
        for model in ['graph', 'transformer', 'deep_sets', 'mil', 'set_transformer']: #'graph', 'transformer', 'deep_sets', 'mil', 'set_transformer'
            for use_coords in [True, False]:
                for use_demo in [True, False]:
                    cfg.use_coords = use_coords
                    cfg.use_demographic = use_demo
                    cfg.tta_method = method
                    cfg.model_name = model
                    fivefold_cv_tta(cfg)
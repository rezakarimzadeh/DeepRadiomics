import torch
from dataloaders.deep_dataloaders import get_dataloaders_deep_learning, get_center2_as_test_loader
from dataloaders.graph_dataloader import get_dataloaders_graph, get_center2_as_test_loader_graph, test_model_percentage_graph
from dataloaders.ml_dataloaders import get_dataloaders_ml, get_classical_test_loader_center2
from models.MIL import RadiomicsMIL
from models.transformer import RadiomicsTransformer
from utils import read_yaml_file, test_model, compute_classification_metrics, save_json, test_model_graph
from pathlib import Path
import pandas as pd
from main import model_generator, data_function_generator_dl
import numpy as np

def test_model_percentage(model, test_loader, node_percentage=None):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                x, y, mask = batch['features'], batch['labels'], batch.get('pad_mask', None)
                if node_percentage is not None:
                    num_nodes = x.shape[1]
                    k = max(1, int(num_nodes * node_percentage))
                    topk_values, topk_indices = torch.topk(torch.rand(num_nodes), k)
                    x = x[:, topk_indices, :]
                    if mask is not None:
                        mask = mask[:, topk_indices]
                logits = model(x.to(device), mask.to(device) if mask is not None else None)

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

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

def evaluate(cfg, fold_index=0, node_percentage=None):
    config_base_dir = './configs'
    model_root = get_model_dir(cfg)
    ckpt_base_dir = model_root / f"fold_{fold_index}"
    model_configs = read_yaml_file(Path(config_base_dir) / f"{cfg.model_name}.yaml") 
    loader_func, _, test_model_func = data_function_generator_dl(cfg)

    _, _, test_loader = loader_func(cfg, fold_index=fold_index)
    
    sample_batch = next(iter(test_loader))
    if cfg.model_name == "graph":
        input_dim = sample_batch.num_node_features
    else:
        input_dim = sample_batch['features'].shape[-1]

    model  = model_generator(cfg.model_name).load_from_checkpoint(
        checkpoint_path=Path(ckpt_base_dir) / 'checkpoints' / 'best.ckpt',
        input_dim=input_dim,
        config=model_configs).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    runs = []
    for i in range(100):
        if cfg.model_name == "graph":
            y_true, y_pred, y_prob = test_model_percentage_graph(model, test_loader, node_percentage)
        else:
            y_true, y_pred, y_prob = test_model_percentage(model, test_loader, node_percentage)

        classification_results = compute_classification_metrics(cfg.model_name, y_true, y_pred, y_prob)


        fold_results = {'info': "different percentage of nodes for inference",
                        'fold': fold_index, **classification_results, "use_demographic": cfg.use_demographic,
                        'use_coords': cfg.use_coords, 'model_name': cfg.model_name, 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
        runs.append(fold_results)
    
    save_json(Path(ckpt_base_dir)/ "node_percentage" / f"node_percentage_{cfg.node_percentage}.json", runs)

def fivefold_cv_node_percentage(args):
    for fold_idx in range(1):
        print(f"Training fold {fold_idx+1}/5")
        if args.model_name in ["transformer", "deep_sets", "mil", "graph", 'set_transformer']:
            evaluate(args, fold_idx, node_percentage=args.node_percentage)
       
        else:
            raise ValueError(f"Model {args.model_name} not recognized.")
        

if __name__ == '__main__':
    class Args:
        model_name = 'transformer'
        data_root = "/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset"
        results_root = "./Results"
        use_coords = True
        use_demographic = False
        batch_size = 32


    cfg = Args()
    for node_percentage in [i/10 for i in range(2,11)]: 
        for model in ['graph']: #'graph', 'transformer', 'deep_sets', 'mil', 'set_transformer'
            for use_coords in [True, False]:
                for use_demo in [True, False]:
                    cfg.use_coords = use_coords
                    cfg.use_demographic = use_demo
                    cfg.node_percentage = node_percentage
                    cfg.model_name = model
                    fivefold_cv_node_percentage(cfg)
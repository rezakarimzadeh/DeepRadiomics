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
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D


def plot_pca_tta_center1_ref(output_aggregated, save_dir, filename="pca_tta_center1_ref.png"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(block):
        y = np.asarray(block["y_true"]).astype(int)
        x = np.asarray(block["embeddings"], dtype=np.float32)
        return x, y

    c1_x, c1_y = _to_numpy(output_aggregated["center1"])
    c2b_x, c2b_y = _to_numpy(output_aggregated["center2_before_tta"])
    c2a_x, c2a_y = _to_numpy(output_aggregated["center2_after_tta"])

    # fit scaler + PCA ONLY on center1
    scaler = StandardScaler().fit(c1_x)
    c1_x_scaled = scaler.transform(c1_x)
    c2b_x_scaled = scaler.transform(c2b_x)
    c2a_x_scaled = scaler.transform(c2a_x)

    pca = PCA(n_components=2).fit(c1_x_scaled)

    z_c1 = pca.transform(c1_x_scaled)
    z_c2_before = pca.transform(c2b_x_scaled)
    z_c2_after = pca.transform(c2a_x_scaled)

    var_exp = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)

    center_colors = {"center1": "#1f77b4", "center2": "#d62728"}
    class_markers = {0: "o", 1: "s"}

    def _scatter(ax, z1, y1, z2, y2, title):
        for cls in [0, 1]:
            ax.scatter(
                z1[y1 == cls, 0], z1[y1 == cls, 1],
                c=center_colors["center1"],
                marker=class_markers[cls],
                s=38, alpha=0.85,
                edgecolors="white", linewidths=0.4
            )
            ax.scatter(
                z2[y2 == cls, 0], z2[y2 == cls, 1],
                c=center_colors["center2"],
                marker=class_markers[cls],
                s=38, alpha=0.85,
                edgecolors="white", linewidths=0.4
            )
        ax.set_title(title, fontsize=12)

    _scatter(
        axes[0],
        z_c1, c1_y,
        z_c2_before, c2b_y,
        "Center 1 reference PCA: before TTA"
    )
    _scatter(
        axes[1],
        z_c1, c1_y,
        z_c2_after, c2a_y,
        "Center 1 reference PCA: after TTA"
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='Class 1',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 1',
               markerfacecolor=center_colors["center1"], markeredgecolor=center_colors["center1"], markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 2',
               markerfacecolor=center_colors["center2"], markeredgecolor=center_colors["center2"], markersize=7),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )

    # fig.suptitle(
    #     "PCA projection using Center 1 as the reference space",
    #     fontsize=13, y=1.02
    # )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = save_dir / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PCA plot to: {out_path}")

def plot_pca_tta(output_aggregated, save_dir, filename="pca_tta_journal.png"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(block):
        y = np.asarray(block["y_true"]).astype(int)
        x = np.asarray(block["embeddings"], dtype=np.float32)
        return x, y

    c1_x, c1_y = _to_numpy(output_aggregated["center1"])
    c2b_x, c2b_y = _to_numpy(output_aggregated["center2_before_tta"])
    c2a_x, c2a_y = _to_numpy(output_aggregated["center2_after_tta"])

    def _run_pca(x1, x2):
        x = np.vstack([x1, x2])
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        z = pca.fit_transform(x)

        var_exp = pca.explained_variance_ratio_ * 100
        return z[:len(x1)], z[len(x1):], var_exp

    z_c1_before, z_c2_before, var_before = _run_pca(c1_x, c2b_x)
    z_c1_after,  z_c2_after,  var_after  = _run_pca(c1_x, c2a_x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    center_colors = {"center1": "#1f77b4", "center2": "#d62728"}
    class_markers = {0: "o", 1: "s"}

    def _scatter(ax, z1, y1, z2, y2, title, var_exp):
        for cls in [0, 1]:
            ax.scatter(
                z1[y1 == cls, 0], z1[y1 == cls, 1],
                c=center_colors["center1"],
                marker=class_markers[cls],
                s=38, alpha=0.85,
                edgecolors="white", linewidths=0.4
            )
            ax.scatter(
                z2[y2 == cls, 0], z2[y2 == cls, 1],
                c=center_colors["center2"],
                marker=class_markers[cls],
                s=38, alpha=0.85,
                edgecolors="white", linewidths=0.4
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)

    _scatter(
        axes[0],
        z_c1_before, c1_y,
        z_c2_before, c2b_y,
        "Center 1 vs Center 2 (before TTA)",
        var_before
    )

    _scatter(
        axes[1],
        z_c1_after, c1_y,
        z_c2_after, c2a_y,
        "Center 1 vs Center 2 (after TTA)",
        var_after
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0',
               markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='Class 1',
               markerfacecolor='gray', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 1',
               markerfacecolor=center_colors["center1"], markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 2',
               markerfacecolor=center_colors["center2"], markersize=7),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        "PCA visualization of feature embeddings before and after test-time adaptation",
        fontsize=13, y=1.02
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = save_dir / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PCA plot to: {out_path}")


def plot_tsne_tta(output_aggregated, save_dir, filename="tsne_tta_journal.png", random_state=42):
    """
    output_aggregated format:
    {
        "center1": {"y_true": [...], "embeddings": [...]},
        "center2_before_tta": {"y_true": [...], "embeddings": [...]},
        "center2_after_tta": {"y_true": [...], "embeddings": [...]}
    }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(block):
        y = np.asarray(block["y_true"]).astype(int)
        x = np.asarray(block["embeddings"], dtype=np.float32)
        return x, y

    c1_x, c1_y = _to_numpy(output_aggregated["center1"])
    c2b_x, c2b_y = _to_numpy(output_aggregated["center2_before_tta"])
    c2a_x, c2a_y = _to_numpy(output_aggregated["center2_after_tta"])

    def _run_tsne(x1, x2):
        x = np.vstack([x1, x2])
        x = StandardScaler().fit_transform(x)   # helps a lot for t-SNE stability
        z = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (len(x) - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=random_state
        ).fit_transform(x)
        return z[:len(x1)], z[len(x1):]

    z_c1_before, z_c2_before = _run_tsne(c1_x, c2b_x)
    z_c1_after,  z_c2_after  = _run_tsne(c1_x, c2a_x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # journal-like styling
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)

    # colors by center, markers by class
    center_colors = {
        "center1": "#1f77b4",   # blue
        "center2": "#d62728",   # red
    }
    class_markers = {
        0: "o",   # dot
        1: "s",   # square
    }

    def _scatter_panel(ax, z1, y1, z2, y2, title):
        for cls in [0, 1]:
            idx1 = y1 == cls
            idx2 = y2 == cls

            ax.scatter(
                z1[idx1, 0], z1[idx1, 1],
                c=center_colors["center1"],
                marker=class_markers[cls],
                s=38, alpha=0.82,
                edgecolors="white", linewidths=0.4
            )
            ax.scatter(
                z2[idx2, 0], z2[idx2, 1],
                c=center_colors["center2"],
                marker=class_markers[cls],
                s=38, alpha=0.82,
                edgecolors="white", linewidths=0.4
            )
        ax.set_title(title, fontsize=12, pad=10)

    _scatter_panel(
        axes[0], z_c1_before, c1_y, z_c2_before, c2b_y,
        "Center 1 vs Center 2 (before TTA)"
    )
    _scatter_panel(
        axes[1], z_c1_after, c1_y, z_c2_after, c2a_y,
        "Center 1 vs Center 2 (after TTA)"
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='Class 1',
               markerfacecolor='gray', markeredgecolor='gray', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 1',
               markerfacecolor=center_colors["center1"], markeredgecolor=center_colors["center1"], markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Center 2',
               markerfacecolor=center_colors["center2"], markeredgecolor=center_colors["center2"], markersize=7),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle("t-SNE visualization of feature embeddings before and after test-time adaptation", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = save_dir / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved t-SNE plot to: {out_path}")

def get_embeddings(model, test_loader):
        model.eval()
        all_embeddings = []
        all_labels = []
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                x, y, mask = batch['features'], batch['labels'], batch.get('pad_mask', None)
                embs = model(x.to(device), mask.to(device), get_embeddings=True)[1]  # Get logits, ignore embeddings for now
                all_embeddings.extend(embs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return all_labels, all_embeddings



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
    get_data_loaders, center2_loader_func, test_model_func = data_function_generator_dl(cfg)
    train_loader, val_loader, test_loader = get_data_loaders(cfg, fold_index=fold_index)

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

    # get embeddings for center1
    center1_y_true, center1_embeddings = get_embeddings(model, test_loader)
    print(f"shape of center1 embeddings: {np.array(center1_embeddings).shape}, gt shape: {np.array(center1_y_true).shape}")
    # get embeddings before TTA for center2
    center2_y_true, center2_embeddings = get_embeddings(model, center2_loader)
    print(f"shape of center2 embeddings: {np.array(center2_embeddings).shape}, gt shape: {np.array(center2_y_true).shape}")

    if cfg.tta_method == 'norm':
        print("Starting Test Time Adaptation with NORM...")
        tta_model = setup_norm(model)
    else:
        print("Starting Test Time Adaptation with TENT...")
        tta_model = setup_tent(model, model_configs['tta_lr'])
    
    # get embeddings after TTA for center2
    center2_y_true_tta, center2_embeddings_tta = get_embeddings(tta_model, center2_loader)
    print(f"shape of center2 embeddings after TTA: {np.array(center2_embeddings_tta).shape}, gt shape: {np.array(center2_y_true_tta).shape}")

    outputs = {
        "center1": {"y_true": center1_y_true, "embeddings": center1_embeddings},
        "center2_before_tta": {"y_true": center2_y_true, "embeddings": center2_embeddings},
        "center2_after_tta": {"y_true": center2_y_true_tta, "embeddings": center2_embeddings_tta}
    }
    return outputs

def fivefold_cv_tta(args):
    output_aggregated = {
        "center1": {"y_true": [], "embeddings": []},
        "center2_before_tta": {"y_true": [], "embeddings": []},
        "center2_after_tta": {"y_true": [], "embeddings": []}
    }
    for fold_idx in range(5):
        print(f"Training fold {fold_idx+1}/5")
        if args.model_name in ["transformer", "deep_sets", "mil", "graph", 'set_transformer']:
            fold_output = evaluate(args, fold_idx)
        else:
            raise ValueError(f"Model {args.model_name} not recognized.")
        for key in output_aggregated.keys():
            output_aggregated[key]["y_true"].extend(np.array(fold_output[key]["y_true"]))
            output_aggregated[key]["embeddings"].extend(np.array(fold_output[key]["embeddings"]))

    # perform tsne on the aggregated embeddings for center2 before and after TTA
    plot_tsne_tta(
        output_aggregated,
        save_dir=Path("notebooks/figures"),
        filename="tsne_center_shift_before_after_tta.png"
    )
    # perform PCA on the aggregated embeddings for center2 before and after TTA
    plot_pca_tta(
        output_aggregated,
        save_dir=Path("notebooks/figures"),
        filename="pca_center_shift_before_after_tta.png"
    )
    # perform PCA using center1 as reference space
    plot_pca_tta_center1_ref(
        output_aggregated,
        save_dir=Path("notebooks/figures"),
        filename="pca_center_shift_before_after_tta_center1_reference.png"
    )


if __name__ == '__main__':
    class Args:
        model_name = 'transformer'
        data_root = "/home/reza/Documents/Reza_projects/08_drarabi_lymphnodes/new_dataset"
        results_root = "./Results"
        use_coords = True
        use_demographic = True
        tta_method = 'tent'
        batch_size = 256
    
    cfg = Args()
    fivefold_cv_tta(cfg)


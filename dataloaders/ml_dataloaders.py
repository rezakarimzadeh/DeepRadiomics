import ast
from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
import numpy as np
import scipy


def read_json(file_path):   
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(data_dict, data_root, use_coords, use_demographic):
    aggregated_data = list()
    for idx in range(len(data_dict)):
        (sample_dir_name, sample_info) = list(data_dict[idx].items())[0]
        hl_label = 0 if sample_info['Binary Label'] == 'NHL' else 1
        radiomics_path = os.path.join(data_root, sample_dir_name, 'radiomics', 'radiomics_each_lesion.json')
        radiomics_features_dict = read_json(radiomics_path)

        aggregated_radiomics = []
        nodes_coordinates_list = list(radiomics_features_dict.keys())
        for node, radiomics_features in radiomics_features_dict.items():
            radiomics_list = list(radiomics_features.values())
            if use_demographic:
                demographic_info = [sample_info['Gender(Male=0, Fmale:1)'], sample_info['Age']]
                radiomics_list = radiomics_list + demographic_info
            if use_coords:
                coords_features = list(ast.literal_eval(node))
                radiomics_list = radiomics_list + coords_features
            features = np.array(radiomics_list)
            aggregated_radiomics.append(features)
        if not aggregated_radiomics:
            # print(f"No valid lesions found for {sample_dir_name}, skipping.")
            continue
        aggregated_radiomics = np.array(aggregated_radiomics)
        skew_vals = scipy.stats.skew(
            aggregated_radiomics, 
            axis=0, 
            nan_policy='omit'
        )

        # Replace remaining NaNs
        skew_vals = np.nan_to_num(skew_vals, nan=0.0)
        set_level_aggrigation = np.concatenate([aggregated_radiomics.mean(axis=0),
                                                # add std deviation
                                                aggregated_radiomics.std(axis=0),
                                                np.median(aggregated_radiomics, axis=0),
                                                aggregated_radiomics.max(axis=0),
                                                aggregated_radiomics.min(axis=0),
                                                # add skewness 
                                                skew_vals,
        ])
        # print(skew_vals)
        if np.isnan(set_level_aggrigation).any():
            print(f"NaN values found in features for {sample_dir_name}, skipping.")
            # print(set_level_aggrigation)
            # continue
        aggregated_data.append({ "features": set_level_aggrigation, "label": hl_label, "ID": sample_dir_name, "nodes_coordinates": nodes_coordinates_list })

    return aggregated_data


def get_masih_data_folds(data_root, fold_index):
    split_filename = os.path.join(data_root, "Kfold_splits", f"masih_5foldCV_split_{fold_index}.json")
    split_data = read_json(split_filename)
    return split_data['train'], split_data['val'], split_data['test']


def get_dataloaders_ml(config, fold_index):
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(config.data_root, fold_index)
    masih_root = os.path.join(config.data_root, "Masih-SUV")
    
    masih_train = masih_train_dict + masih_val_dict
    masih_aggregated_data_dicts = prepare_data(masih_train, masih_root, config.use_coords, config.use_demographic)
    train_data = [d['features'] for d in masih_aggregated_data_dicts]
    train_labels = [d['label'] for d in masih_aggregated_data_dicts]
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_labels)

    masih_test_aggregated_data_dicts = prepare_data(masih_test_dict, masih_root, config.use_coords, config.use_demographic)
    test_data = [d['features'] for d in masih_test_aggregated_data_dicts]
    test_labels = [d['label'] for d in masih_test_aggregated_data_dicts]
    X_test = np.asarray(test_data)
    y_test = np.asarray(test_labels)
    return X_train, y_train, X_test, y_test

def get_classical_test_loader_center2(config):
    test_dataset = read_json(os.path.join(config.data_root, "dataset_directories.json"))
    test_dataset = test_dataset.get("Razavi-SUV", [])
    center2_root = os.path.join(config.data_root, "Razavi-SUV")
    center2_aggregated_data_dicts = prepare_data(test_dataset, center2_root, config.use_coords, config.use_demographic)
    train_data = [d['features'] for d in center2_aggregated_data_dicts]
    train_labels = [d['label'] for d in center2_aggregated_data_dicts]
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_labels)
    return X_train, y_train

# ================= Dataloaders with statistics for dl ================= #
class CustomDataset(Dataset):
    def __init__(self, data_root, data_dict, use_coords):
        self.data_root = data_root
        self.data_dict = data_dict
        self.aggregated_data = prepare_data(data_dict, data_root, use_coords)

    def __len__(self):
        return len(self.aggregated_data)
    
    def __getitem__(self, idx):
        data_dict = self.aggregated_data[idx]
        features = torch.tensor(np.array(data_dict['features']), dtype=torch.float32)
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        nodes_coordinates = data_dict['nodes_coordinates']
        pid = data_dict['ID']
        output = {"pids": pid, "nodes_coordinates": nodes_coordinates, 
              "features": features, "labels": label}
        return output
    
def get_classical_dataloaders(config, fold_index):
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(config.data_root, fold_index)
    masih_root = os.path.join(config.data_root, "Masih-SUV")
    print(f"Train size: {len(masih_train_dict)}, Val size: {len(masih_val_dict)}, Test size: {len(masih_test_dict)}")
    train_dataset = CustomDataset(masih_root, masih_train_dict, config.use_coords)
    val_dataset = CustomDataset(masih_root, masih_val_dict, config.use_coords)
    test_dataset = CustomDataset(masih_root, masih_test_dict, config.use_coords)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader

# ================= demographic features ================= #

def prepare_demographic_data(data_dict, data_root, use_age, use_gender):
    aggregated_data = list()
    for idx in range(len(data_dict)):
        (sample_dir_name, sample_info) = list(data_dict[idx].items())[0]
        hl_label = 0 if sample_info['Binary Label'] == 'NHL' else 1
        radiomics_path = os.path.join(data_root, sample_dir_name, 'radiomics', 'radiomics_each_lesion.json')
        radiomics_features_dict = read_json(radiomics_path)

        aggregated_features = []
        nodes_coordinates_list = list(radiomics_features_dict.keys())
        for node, radiomics_features in radiomics_features_dict.items():
            if use_age:
                aggregated_features.append(sample_info['Age'])
            if use_gender:
                aggregated_features.append(sample_info['Gender(Male=0, Fmale:1)'])
            break  # We only need to add demographic features once per patient, so we break after the first lesion
        if not aggregated_features:
            # print(f"No valid lesions found for {sample_dir_name}, skipping.")
            continue
        aggregated_features = np.array(aggregated_features)
        
        aggregated_data.append({ "features": aggregated_features, "label": hl_label, "ID": sample_dir_name})#), "nodes_coordinates": nodes_coordinates_list })

    return aggregated_data

def get_demographic_ml_center1(config, fold_index):
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(config.data_root, fold_index)
    masih_root = os.path.join(config.data_root, "Masih-SUV")
    
    masih_train = masih_train_dict + masih_val_dict
    masih_aggregated_data_dicts = prepare_demographic_data(masih_train, masih_root, config.use_age, config.use_gender)
    train_data = [d['features'] for d in masih_aggregated_data_dicts]
    train_labels = [d['label'] for d in masih_aggregated_data_dicts]
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_labels)

    masih_test_aggregated_data_dicts = prepare_demographic_data(masih_test_dict, masih_root, config.use_age, config.use_gender)
    test_data = [d['features'] for d in masih_test_aggregated_data_dicts]
    test_labels = [d['label'] for d in masih_test_aggregated_data_dicts]
    X_test = np.asarray(test_data)
    y_test = np.asarray(test_labels)
    return X_train, y_train, X_test, y_test

def get_demographic_ml_center2(config):
    test_dataset = read_json(os.path.join(config.data_root, "dataset_directories.json"))
    test_dataset = test_dataset.get("Razavi-SUV", [])
    center2_root = os.path.join(config.data_root, "Razavi-SUV")
    center2_aggregated_data_dicts = prepare_demographic_data(test_dataset, center2_root, config.use_age, config.use_gender)
    train_data = [d['features'] for d in center2_aggregated_data_dicts]
    train_labels = [d['label'] for d in center2_aggregated_data_dicts]
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_labels)
    return X_train, y_train

# ================= domain shift study data ================= #
def get_aggregated_data_domain_shift_study(config, fold_index=0):
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(config.data_root, fold_index)
    masih_root = os.path.join(config.data_root, "Masih-SUV")
    
    center_1 = masih_train_dict + masih_val_dict + masih_test_dict
    center1_data_dicts = prepare_data(center_1, masih_root, config.use_coords, config.use_demographic)
    center1_data = [d['features'] for d in center1_data_dicts]
    center1_labels = [d['label'] for d in center1_data_dicts]

    c2_dataset = read_json(os.path.join(config.data_root, "dataset_directories.json"))
    center2_dataset = c2_dataset.get("Razavi-SUV", [])
    center2_root = os.path.join(config.data_root, "Razavi-SUV")
    center2_aggregated_data_dicts = prepare_data(center2_dataset, center2_root, config.use_coords, config.use_demographic)
    center2_data = [d['features'] for d in center2_aggregated_data_dicts]
    center2_labels = [d['label'] for d in center2_aggregated_data_dicts]

    output = {
        "center1": {
            "data": np.asarray(center1_data),
            "labels": np.asarray(center1_labels)
        },
        "center2": {
            "data": np.asarray(center2_data),
            "labels": np.asarray(center2_labels)
        }
    }
    return output

from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
import numpy as np


def read_json(file_path):   
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(data_dict, data_root, use_tda):
    aggregated_data = list()
    for idx in range(len(data_dict)):
        (sample_dir_name, sample_info) = list(data_dict[idx].items())[0]
        hl_label = 0 if sample_info['Binary Label'] == 'NHL' else 1
        radiomics_path = os.path.join(data_root, sample_dir_name, 'radiomics', 'radiomics_each_lesion.json')
        radiomics_features_dict = read_json(radiomics_path)
        if use_tda:
            tda_path = os.path.join(data_root, sample_dir_name, 'radiomics', 'tda_features_each_lesion.json')
            if not os.path.exists(tda_path):
                continue
            tda_features_dict = read_json(tda_path)

        aggregated_radiomics = []
        nodes_coordinates_list = list(radiomics_features_dict.keys())
        for node, radiomics_features in radiomics_features_dict.items():
            radiomics_list = list(radiomics_features.values())
            demographic_info = [sample_info['Gender(Male=0, Fmale:1)'], sample_info['Age'], sample_info['Stage']]
            if use_tda:
                tda_features = list(tda_features_dict.get(node, [0]*8))
                radiomics_list = radiomics_list + tda_features
            features = np.array(radiomics_list + demographic_info)
            aggregated_radiomics.append(features)
        if not aggregated_radiomics:
            print(f"No valid lesions found for {sample_dir_name}, skipping.")
            continue
        aggregated_radiomics = np.array(aggregated_radiomics)
        set_level_aggrigation = np.concatenate([aggregated_radiomics.mean(axis=0),
                                                aggregated_radiomics.std(axis=0),
                                                np.median(aggregated_radiomics, axis=0),
                                                aggregated_radiomics.max(axis=0),
                                                aggregated_radiomics.min(axis=0)])
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
    masih_aggregated_data_dicts = prepare_data(masih_train, masih_root, config.use_tda)
    train_data = [d['features'] for d in masih_aggregated_data_dicts]
    train_labels = [d['label'] for d in masih_aggregated_data_dicts]
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_labels)

    masih_test_aggregated_data_dicts = prepare_data(masih_test_dict, masih_root, config.use_tda)
    test_data = [d['features'] for d in masih_test_aggregated_data_dicts]
    test_labels = [d['label'] for d in masih_test_aggregated_data_dicts]
    X_test = np.asarray(test_data)
    y_test = np.asarray(test_labels)
    return X_train, y_train, X_test, y_test

# ================= Dataloaders with statistics for dl ================= #
class CustomDataset(Dataset):
    def __init__(self, data_root, data_dict, use_tda):
        self.data_root = data_root
        self.data_dict = data_dict
        self.aggregated_data = prepare_data(data_dict, data_root, use_tda)

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
    train_dataset = CustomDataset(masih_root, masih_train_dict, config.use_tda)
    val_dataset = CustomDataset(masih_root, masih_val_dict, config.use_tda)
    test_dataset = CustomDataset(masih_root, masih_test_dict, config.use_tda)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


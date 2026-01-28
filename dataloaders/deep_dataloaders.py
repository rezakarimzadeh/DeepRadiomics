import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ast 


def read_json(file_path):   
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class CustomDataset(Dataset):
    def __init__(self, data_root, data_dict, train, use_coords, use_demographic):
        '''
        data_root: root directory containing sample folders
        data_dict: list of dicts, each dict contains {sample_dir_name: sample_info}
        train: bool, whether in training mode (for data augmentation)
        use_coords: bool, whether to use TDA features along with radiomics
        '''
        self.data_root = data_root
        self.data_dict = data_dict
        self.train = train
        self.use_demographic = use_demographic
        self.aggregated_data = self.prepare_data(data_dict, data_root, use_coords, use_demographic)

    def __len__(self):
        return len(self.aggregated_data)

    def prepare_data(self, data_dict, data_root, use_coords, use_demographic):
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
            aggregated_data.append({ "features": aggregated_radiomics, "label": hl_label, "ID": sample_dir_name, "nodes_coordinates": nodes_coordinates_list })
        return aggregated_data

    def __getitem__(self, idx):
        data_dict = self.aggregated_data[idx]
        features = torch.tensor(np.array(data_dict['features']), dtype=torch.float32)
        label = torch.tensor(data_dict['label'], dtype=torch.long)
        nodes_coordinates = data_dict['nodes_coordinates']
        pid = data_dict['ID']
        # select random number of permutaed features during training 
        if self.train:
            num_lesions = features.size(0)
            permutation = torch.randperm(num_lesions)
            random_idx = torch.randint(1, num_lesions+1, (1,)).item()
            features = features[permutation[:random_idx], :]
            nodes_coordinates = [nodes_coordinates[i] for i in permutation[:random_idx].tolist()]
        
        return features, label, nodes_coordinates, pid
    

def collate_fn(batch):
    features = [item[0] for item in batch]          # each: [Ti, F]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    nodes_coordinates = [item[2] for item in batch]
    pids = [item[3] for item in batch]

    max_lesions = max(f.shape[0] for f in features)
    F = features[0].shape[1]

    padded_features = []
    pad_masks = []                                   # [B, T], True where PAD
    for f in features:
        T = f.shape[0]
        pad_size = max_lesions - T
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, F), dtype=f.dtype)
            # repeat features as padding
            # pad_tensor = f[torch.arange(pad_size) % T]
            padded_f = torch.cat([f, pad_tensor], dim=0)
            pad_mask = torch.zeros(max_lesions, dtype=torch.bool)
            pad_mask[T:] = True
        else:
            padded_f = f
            pad_mask = torch.zeros(max_lesions, dtype=torch.bool)
        padded_features.append(padded_f)
        pad_masks.append(pad_mask)

    features_tensor = torch.stack(padded_features, dim=0) 
    pad_mask_tensor = torch.stack(pad_masks, dim=0)       
    output = {"pids": pids, "nodes_coordinates": nodes_coordinates, 
              "features": features_tensor, "pad_mask": pad_mask_tensor,
              "labels": labels}
    return output


def get_masih_data_folds(data_root, fold_index):
    split_filename = os.path.join(data_root, "Kfold_splits", f"masih_5foldCV_split_{fold_index}.json")
    split_data = read_json(split_filename)
    return split_data['train'], split_data['val'], split_data['test']


def get_dataloaders_deep_learning(cfg, fold_index):    
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(cfg.data_root, fold_index)
    masih_root = os.path.join(cfg.data_root, "Masih-SUV")
    print(f"Train size: {len(masih_train_dict)}, Val size: {len(masih_val_dict)}, Test size: {len(masih_test_dict)}")
    train_dataset = CustomDataset(masih_root, masih_train_dict, train=True, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    val_dataset = CustomDataset(masih_root, masih_val_dict, train=False, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    test_dataset = CustomDataset(masih_root, masih_test_dict, train=False, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def get_center2_as_test_loader(cfg):
    test_dataset = read_json(os.path.join(cfg.data_root, "dataset_directories.json"))
    center2_root = os.path.join(cfg.data_root, "Razavi-SUV")
    center2_test_dict = test_dataset.get("Razavi-SUV", [])
    print(f"Razavi Test size: {len(center2_test_dict)}")
    test_dataset = CustomDataset(center2_root, center2_test_dict, train=False, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    return test_loader
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import os
import json
import ast

def read_json(file_path):   
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class Sequence2GraphDataset(Dataset):
    def __init__(self, data_root, data_dict, use_coords, use_demographic):
        '''
        data_root: root directory containing sample folders
        data_dict: list of dicts, each dict contains {sample_dir_name: sample_info}
        train: bool, whether in training mode (for data augmentation)
        use_coords: bool, whether to use TDA features along with radiomics
        '''
        self.data_root = data_root
        self.data_dict = data_dict
        self.aggregated_data = self.prepare_data(data_dict, data_root, use_coords, use_demographic)
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx):
        return self.pyg_dataset[idx]
    
    def prepare_data(self, data_dict, data_root, use_coords, use_demographic):
        self.pyg_dataset = []
        self.aggregated_data = list()
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
            seq = torch.tensor(np.array(aggregated_radiomics), dtype=torch.float32)
            label = torch.tensor(hl_label, dtype=torch.float32)
            edge_weight, adjacency_matrix = self.edge_weight_function(nodes_coordinates_list)
            graph_data = Data(x=seq, edge_index=None, y=label, edge_attr=edge_weight)
            graph_data.edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t().contiguous()
            # graph_data.edge_attr = torch.nonzero(edge_weight, as_tuple=False).t().contiguous()
            self.pyg_dataset.append(graph_data)
            self.aggregated_data.append({ "features": aggregated_radiomics, "label": hl_label, "ID": sample_dir_name, "nodes_coordinates": nodes_coordinates_list })
    
    def edge_weight_function(self, nodes_coordinates_list, x_size=224, y_size=224, z_size=335):
        ''''
        This function generates the connection weights between nodes of the given sequence based on:
            1. normalized euclidean distance between nodes   
                wd = exp(-||d||)
            2. order of the nodes
        
        inputs:
            x_coordinates: fixation points x coordinates
            y_coordinates: fixation points y coordinates
            w: image width
            h: image height
        output:
            edge weights tensor
            adjacancy_matrix
        '''
        nodes_coordinates_list = [np.array(ast.literal_eval(coord)) for coord in nodes_coordinates_list]
        number_of_nodes = len(nodes_coordinates_list)
        _distance = torch.zeros([number_of_nodes, number_of_nodes])
        for i, (x1, y1, z1) in enumerate(nodes_coordinates_list):
            for j, (x2, y2, z2) in enumerate(nodes_coordinates_list):
                dxyz = ((x1-x2)/x_size)**2 + ((y1-y2)/y_size)**2 + ((z1-z2)/z_size)**2
                _distance[i, j] = dxyz

        mask = _distance!=0 
        _distance = torch.sqrt(_distance)
        distance_weights = torch.exp(-_distance*10)*mask

        edge_weights = distance_weights
        adjacency_matrix = edge_weights !=0 
        flattened_edge_weights = edge_weights[adjacency_matrix].view(-1,1)
        return flattened_edge_weights, adjacency_matrix
    

def get_masih_data_folds(data_root, fold_index):
    split_filename = os.path.join(data_root, "Kfold_splits", f"masih_5foldCV_split_{fold_index}.json")
    split_data = read_json(split_filename)
    return split_data['train'], split_data['val'], split_data['test']


def get_dataloaders_graph(cfg, fold_index):    
    masih_train_dict, masih_val_dict, masih_test_dict = get_masih_data_folds(cfg.data_root, fold_index)
    masih_root = os.path.join(cfg.data_root, "Masih-SUV")
    print(f"Train size: {len(masih_train_dict)}, Val size: {len(masih_val_dict)}, Test size: {len(masih_test_dict)}")
    train_dataset = Sequence2GraphDataset(masih_root, masih_train_dict, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    val_dataset = Sequence2GraphDataset(masih_root, masih_val_dict, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    test_dataset = Sequence2GraphDataset(masih_root, masih_test_dict, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def get_center2_as_test_loader_graph(cfg):
    test_dataset = read_json(os.path.join(cfg.data_root, "dataset_directories.json"))
    center2_root = os.path.join(cfg.data_root, "Razavi-SUV")
    center2_test_dict = test_dataset.get("Razavi-SUV", [])
    print(f"Razavi Test size: {len(center2_test_dict)}")
    test_dataset = Sequence2GraphDataset(center2_root, center2_test_dict, use_coords=cfg.use_coords, use_demographic=cfg.use_demographic)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    return test_loader
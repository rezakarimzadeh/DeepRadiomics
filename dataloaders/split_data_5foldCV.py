import json
import os
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedShuffleSplit

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def print_label_distribution(data, label_key='Binary Label'):
    label_counts = {}
    for item in data:
        val = list(item.values())[0]
        label = val['Binary Label']
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")
        print(f"  Label {label}: {count / len(data) * 100:.2f}%")

def split_data_5foldCV(data_root, output_dir, seed=11):
    all_data = load_from_json(os.path.join(data_root, "dataset_directories.json"))
    masih_data = all_data.get("Masih-SUV", [])
    print(f"Total Masih-SUV samples: {len(masih_data)}")
    print_label_distribution(masih_data)
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(masih_data)):
        train_val_data = [masih_data[idx] for idx in train_index]
        test_data = [masih_data[idx] for idx in test_index]
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.2, random_state=seed
        )
        split_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        print(f"Fold {i+1}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        print("train label distribution: ")
        print_label_distribution(train_data)
        print("val label distribution: ")
        print_label_distribution(val_data)
        print("test label distribution: ")
        print_label_distribution(test_data)

        split_filename = os.path.join(output_dir, "Kfold_splits", f"masih_5foldCV_split_{i}.json")
        os.makedirs(os.path.dirname(split_filename), exist_ok=True)
        save_to_json(split_dict, split_filename)
        print(f"Saved fold {i+1} split to {split_filename}")

def get_labels(data, label_key='Binary Label'):
    labels = []
    for item in data:
        val = list(item.values())[0]
        label = val[label_key]
        labels.append(label)
    return labels

def stratified_split_data_5foldCV(data_root, output_dir, seed=42):
    all_data = load_from_json(os.path.join(data_root, "dataset_directories.json"))
    masih_data = all_data.get("Masih-SUV", [])
    labels = get_labels(masih_data)
    print(f"Total Masih-SUV samples: {len(masih_data)}")
    print_label_distribution(masih_data)
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(sss.split(masih_data, labels)):
        train_val_data = [masih_data[idx] for idx in train_index]
        test_data = [masih_data[idx] for idx in test_index]
        train_val_labels = get_labels(train_val_data)
        # stratified split train and val with 75%-25% ratio
        tv_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_indices, val_indices = next(tv_sss.split(train_val_data, train_val_labels))
        train_data = [train_val_data[idx] for idx in train_indices]
        val_data = [train_val_data[idx] for idx in val_indices]
        split_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        print(f"Fold {i+1}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        print("train label distribution: ")
        print_label_distribution(train_data)
        print("val label distribution: ")
        print_label_distribution(val_data)
        print("test label distribution: ")
        print_label_distribution(test_data)

        split_filename = os.path.join(output_dir, "Kfold_splits", f"masih_5foldCV_split_{i}.json")
        os.makedirs(os.path.dirname(split_filename), exist_ok=True)
        save_to_json(split_dict, split_filename)
        print(f"Saved fold {i+1} split to {split_filename}")

if __name__ == "__main__":
    data_root = "../../new_dataset"
    # split_data_5foldCV(data_root, data_root)
    stratified_split_data_5foldCV(data_root, data_root)
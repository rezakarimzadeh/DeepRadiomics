import os
import pandas as pd
import json
import re
import numpy as np

def match_func(folder_name: str, clinical_data: pd.DataFrame) -> dict:
    """
    Match folder name to clinical data patient ID.
    Returns the matched row as a dict, or empty dict if no match.
    """
    norm_folder = folder_name.strip().lower().replace("_", " ")
    norm_folder = re.sub(r'[-\s]*\d+\s*$', '', norm_folder).strip()
    # norm_folder = norm_folder.replace(" ", "")
    # print(f"Normalized folder name for matching: {norm_folder}")

    for _, row in clinical_data.iterrows():
        pid = str(row['patient_id']).strip().lower().replace("_", " ")
        # print(pid)
        if pid in norm_folder:
            return row.to_dict()
        if norm_folder in pid:
            return row.to_dict()
        # print(f"No match for folder: {norm_folder}, patient IDs: {pid}")
    # exit()
    return {}


def create_dataset_dirs(
        root_dir: str,
        hospital_list: list=['Masih-SUV', 'Razavi-SUV'],
        clinical_data_name: str="biopsy_info.xlsx",
):
    clinical_data = pd.read_excel(os.path.join(root_dir, clinical_data_name))
    patient_count = 0
    dir_dict = {hospital: [] for hospital in hospital_list}
    for hospital in hospital_list:
        hospital_path = os.path.join(root_dir, hospital)
        if not os.path.exists(hospital_path):
            continue
        patient_dirs = [d for d in os.listdir(hospital_path) if os.path.isdir(os.path.join(hospital_path, d))]
        print(f"Processing hospital: {hospital}, total patient dirs: {len(patient_dirs)}")
        for pid in patient_dirs:
            matched_info = match_func(pid, clinical_data)
            if not matched_info:
                print(f"No clinical data match for patient ID: {pid}")
                continue
            nii_file = [f for f in os.listdir(os.path.join(hospital_path, pid)) if f.endswith(".nii")][0]
            nrrd_file = [f for f in os.listdir(os.path.join(hospital_path, pid)) if f.endswith(".nrrd")][0]
            dir_dict[hospital].append({
                pid: {
                    "image": nii_file,
                    "segmentation": nrrd_file,
                    **matched_info
                }
            })
            patient_count += 1
    print(f"Total matched patients: {patient_count}")

    with open(os.path.join(root_dir, "dataset_directories.json"), "w") as f:
        json.dump(dir_dict, f, indent=4)


def split_datasets(
        dataset_json_path: str,
        train_ratio: float=0.7,
        val_ratio: float=0.1,
        test_ratio: float=0.2,
        random_seed: int=42
):
    with open(dataset_json_path, "r") as f:
        dir_dict = json.load(f)

    np.random.seed(random_seed)
    train_set, val_set, test_set = {}, {}, {}
    for hospital, patients in dir_dict.items():
        print(f"Splitting dataset for hospital: {hospital}, total patients: {len(patients)}")
        np.random.shuffle(patients)
        total_count = len(patients)
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        train_set[hospital] = patients[:train_end]
        val_set[hospital] = patients[train_end:val_end]
        test_set[hospital] = patients[val_end:]

    base_path = os.path.dirname(dataset_json_path)
    with open(os.path.join(base_path, "train_dataset.json"), "w") as f:
        json.dump(train_set, f, indent=4)
    with open(os.path.join(base_path, "val_dataset.json"), "w") as f:
        json.dump(val_set, f, indent=4)
    with open(os.path.join(base_path, "test_dataset.json"), "w") as f:
        json.dump(test_set, f, indent=4)

if __name__ == "__main__":
    data_root = "../../new_dataset"
    create_dataset_dirs(data_root)
    split_datasets(os.path.join(data_root, "dataset_directories.json"))

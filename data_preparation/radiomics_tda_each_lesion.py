import SimpleITK as sitk    
import numpy as np
import os
from tqdm import tqdm
from radiomics import featureextractor
import json


import gudhi as gd

def bbox_from_mask(mask):
    """Get tight bounding box of non-zero region in a 3D mask."""
    coords = np.argwhere(mask > 0)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1  # +1 because slices are exclusive
    return (zmin, zmax), (ymin, ymax), (xmin, xmax)

def summarize_diagram(diag):
    """
    diag: array-like of shape [N, 2] with (birth, death)
    ignores bars with infinite death.
    """
    if len(diag) == 0:
        return dict(num=0, total_pers=0.0, max_pers=0.0, mean_pers=0.0)

    # keep only finite bars
    finite = [(b, d) for (b, d) in diag if np.isfinite(d) and np.isfinite(b)]
    if len(finite) == 0:
        return dict(num=0, total_pers=0.0, max_pers=0.0, mean_pers=0.0)

    lengths = np.array([d - b for (b, d) in finite], dtype=float)

    return dict(
        num=len(lengths),
        total_pers=float(lengths.sum()),
        max_pers=float(lengths.max()),
        mean_pers=float(lengths.mean()),
    )

def tda_features_for_label(pet, mask, label):
    """
    Compute simple TDA features for one node label.
    pet:  3D numpy array (PET)
    mask: 3D numpy array (labels)
    label: int (node label in mask)
    """
    # isolate this node
    node_mask = (mask == label).astype(np.uint8)
    if node_mask.sum() == 0:
        return None  # empty

    # crop to bounding box to speed things up
    (z0, z1), (y0, y1), (x0, x1) = bbox_from_mask(node_mask)
    pet_crop = pet[z0:z1, y0:y1, x0:x1]
    mask_crop = node_mask[z0:z1, y0:y1, x0:x1]

    # build filtration: outside node = large value so they appear late
    max_val = float(pet_crop.max())
    big_val = max_val + 1.0
    filt = np.where(mask_crop > 0, pet_crop, big_val)

    # Gudhi expects a flat array for top_dimensional_cells
    cc = gd.CubicalComplex(
        dimensions=filt.shape,
        top_dimensional_cells=filt.ravel()
    )
    cc.compute_persistence()

    # get diagrams for H0 and H1
    diag0 = cc.persistence_intervals_in_dimension(0)
    diag1 = cc.persistence_intervals_in_dimension(1)

    # summarize
    h0 = summarize_diagram(diag0)
    h1 = summarize_diagram(diag1)

    # return as a flat feature vector (you can change this to a dict if you prefer)
    features = np.array([
        h0["num"], h0["total_pers"], h0["max_pers"], h0["mean_pers"],
        h1["num"], h1["total_pers"], h1["max_pers"], h1["mean_pers"],
    ], dtype=float)

    return features
#  ======================================================================

EXTRACTOR = featureextractor.RadiomicsFeatureExtractor("pet_radiomics.yaml")


def read_sitk_image(file_path):
    img = sitk.ReadImage(file_path)
    return img

def sitk_image_to_array(sitk_image):
    array = sitk.GetArrayFromImage(sitk_image)
    return array

def array_to_sitk_image(array, reference_sitk_image):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.CopyInformation(reference_sitk_image)
    return sitk_image

def get_voxel_volume(sitk_image):
    spacing = sitk_image.GetSpacing()
    return spacing[0] * spacing[1] * spacing[2]

def read_json(file_path):   
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_segmentation_volume_ml(np_image, voxel_volume):
    return np.sum(np_image > 0) * voxel_volume / 1000.0  # convert to ml


def make_serializable(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    else:
        return obj

def filter_radiomics_features(features_dict):
    """Keep only the actual radiomics features, not diagnostics."""
    filtered = {
        k: v for k, v in features_dict.items()
        if not k.startswith("diagnostics_")
    }
    return filtered

def extract_radiomics(pet_itk, mask_itk):
    result = EXTRACTOR.execute(pet_itk, mask_itk)
    center_of_mass_index = result['diagnostics_Mask-original_CenterOfMassIndex']
    center_of_mass_index = tuple(float(c) for c in center_of_mass_index)
    result = filter_radiomics_features(result)
    feats = make_serializable(result)
    return feats, center_of_mass_index

def clean_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            clean_dict(v)
        elif isinstance(v, complex):
            d[k] = v.real
        elif isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d

def process_each_lesion_seg_mask(dataset_dicts, data_root):
    for item in tqdm(dataset_dicts):
        sample_dir_name, sample_info = list(item.items())[0]
        img_name = sample_info['image']
        img_dir = os.path.join(data_root, sample_dir_name, img_name.replace('.nii', '_prep_img.nii.gz'))
        seg_name = sample_info['segmentation']
        seg_dir = os.path.join(data_root, sample_dir_name, seg_name.replace('.nrrd', '_prep_seg.nii.gz'))

        img_sitk = read_sitk_image(img_dir)
        seg_sitk = read_sitk_image(seg_dir)

        img_array = sitk_image_to_array(img_sitk)
        seg_array = sitk_image_to_array(seg_sitk)

        voxel_volume = get_voxel_volume(seg_sitk)
        
        radiomics_dict = {}
        tda_features_dict = {}
        unique_labels = np.unique(seg_array)
        for label in unique_labels:
            if label == 0:
                continue  # skip background
            # Create binary mask for the current label
            label_mask = (seg_array == label).astype(np.uint8)
            volume_ml = get_segmentation_volume_ml(label_mask, voxel_volume)
            if volume_ml < 1.0:
                continue  # skip small volumes
            
            unified_mask_sitk = array_to_sitk_image(label_mask, seg_sitk)
            radiomics_features, center_of_mass_index = extract_radiomics(img_sitk, unified_mask_sitk)
            radiomics_dict[str(center_of_mass_index)] = radiomics_features

            tda_features = tda_features_for_label(img_array, seg_array, label)
            if tda_features is not None:
                tda_features_dict[str(center_of_mass_index)] = tda_features.tolist()
        # Save the new unified mask
        if not radiomics_dict:
            print(f"No valid lesions found for {sample_dir_name}, skipping.")
            continue

        radiomics_dict = clean_dict(radiomics_dict)
        output_radiomics_path = os.path.join(data_root, sample_dir_name, "radiomics", "radiomics_each_lesion.json")
        output_tda_path = os.path.join(data_root, sample_dir_name, "radiomics", "tda_features_each_lesion.json")

        os.makedirs(os.path.dirname(output_radiomics_path), exist_ok=True)
        with open(output_radiomics_path, 'w') as f:
            json.dump(radiomics_dict, f, indent=4)
        
        with open(output_tda_path, 'w') as f:
            json.dump(tda_features_dict, f, indent=4)
        print(f"Saved radiomics features to {output_radiomics_path}")


def process_dataset(root_dir):
    data_directories_dir = os.path.join(root_dir, "dataset_directories.json")
    dict_dataset = read_json(data_directories_dir)
    
    masih_root = os.path.join(root_dir, "Masih-SUV")
    razavi_root = os.path.join(root_dir, "Razavi-SUV")

    masih_dicts = dict_dataset.get("Masih-SUV", [])
    razavi_dicts = dict_dataset.get("Razavi-SUV", [])
    process_each_lesion_seg_mask(masih_dicts, masih_root)
    process_each_lesion_seg_mask(razavi_dicts, razavi_root)

if __name__ == "__main__":
    data_root = "../../new_dataset"
    process_dataset(data_root)
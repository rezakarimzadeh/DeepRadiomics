from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, MapTransform,
    SpatialPadd, CenterSpatialCropd, EnsureTyped, SaveImaged, ResizeWithPadOrCropd
)
import os
from tqdm import tqdm

class ImageSegmentationEqualSize(MapTransform):
    """
    Convert labels to one class segmentation

    """
    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        seg = d["segmentation"]
        img_z_size = img.shape[-1]
        seg_z_size = seg.shape[-1]
        print(f"image size: {img.shape}, segmentation size: {seg.shape}")
        if img_z_size > seg_z_size:
            d["image"] = img[..., :seg_z_size]
        elif seg_z_size > img_z_size:
            d["segmentation"] = seg[..., :img_z_size]
        print(f"After equalizing sizes -> image size: {d['image'].shape}, segmentation size: {d['segmentation'].shape}")
        return d

def get_preprocess_transform(output_dir):  # ← add param
    target_spacing = (2.0, 2.0, 3.27)
    out_hw = (224, 224)

    preprocess = Compose([
        LoadImaged(keys=["image", "segmentation"]),
        EnsureChannelFirstd(keys=["image", "segmentation"]),
        Orientationd(keys=["image", "segmentation"], axcodes="RAS"),
        Spacingd(keys=["image", "segmentation"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        SpatialPadd(keys=["image", "segmentation"], spatial_size=(out_hw[0], out_hw[1], -1), mode="constant", constant_values=0),
        CenterSpatialCropd(keys=["image", "segmentation"], roi_size=(out_hw[0], out_hw[1], -1)),
        # ResizeWithPadOrCropd(keys=["image", "segmentation"], spatial_size=None),
        EnsureTyped(keys=["image", "segmentation"]),
        ImageSegmentationEqualSize(keys=["image", "segmentation"]),
        # Save in the provided output_dir (same as input image folder)
        SaveImaged(
            keys="image",
            meta_keys="image_meta_dict",
            output_postfix="prep_img",
            output_ext=".nii.gz",
            output_dir=output_dir,
            separate_folder=False,
            resample=False
        ),
        SaveImaged(
            keys="segmentation",
            meta_keys="segmentation_meta_dict",
            output_postfix="prep_seg",
            output_ext=".nii.gz",
            output_dir=output_dir,
            separate_folder=False,
            resample=False
        ),
    ])
    return preprocess

def perform_preprocessing(data_dicts, data_root):
    for item in tqdm(data_dicts):
        sample_dir_name, sample_info = list(item.items())[0]
        img_name = sample_info['image']
        img_dir = os.path.join(data_root, sample_dir_name, img_name)
        seg_name = sample_info['segmentation']
        seg_dir = os.path.join(data_root, sample_dir_name, seg_name)

        # Build transform with the *image* folder as output_dir
        out_dir = os.path.dirname(img_dir)
        preprocess = get_preprocess_transform(out_dir)

        data = {"image": img_dir, "segmentation": seg_dir}
        preprocess(data)
        # print(img_dir, "-> Preprocessed")


def read_json(file_path):   
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

    
def preprocess_data(data_root):
    dict_dataset = read_json(os.path.join(data_root, "dataset_directories.json"))
    
    masih_root = os.path.join(data_root, "Masih-SUV")
    razavi_root = os.path.join(data_root, "Razavi-SUV")

    masih_dicts = dict_dataset.get("Masih-SUV", [])
    razavi_dicts = dict_dataset.get("Razavi-SUV", [])


    perform_preprocessing(masih_dicts, masih_root)
    perform_preprocessing(razavi_dicts, razavi_root)

if __name__ == "__main__":
    data_root = "../../new_dataset"
    preprocess_data(data_root)
    
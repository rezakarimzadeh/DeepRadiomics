import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# -----------------------------
# Existing helpers (keep yours)
# -----------------------------
def read_sitk_image(file_path):
    img = sitk.ReadImage(file_path)
    return img

def sitk_image_to_array(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)

def get_voxel_volume(sitk_image):
    spacing = sitk_image.GetSpacing()
    return spacing[0] * spacing[1] * spacing[2]   # mm^3

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_segmentation_volume_ml(np_image, voxel_volume):
    return np.sum(np_image > 0) * voxel_volume / 1000.0  # mm^3 -> mL


# -----------------------------
# New minimal biomarker helpers
# -----------------------------
# def compute_patient_pet_biomarkers(img_array, seg_array, voxel_volume):
#     """
#     Minimal implementation:
#       - lesion MTV = lesion mask volume
#       - patient TMTV = sum of lesion MTVs
#       - patient SUVmax = max SUV across all lesion voxels
#     """
#     unique_labels = np.unique(seg_array)

#     suvmax_patient = None
#     lesion_mtvs = []

#     for label in unique_labels:
#         if label == 0:
#             continue

#         label_mask = (seg_array == label)
#         volume_ml = get_segmentation_volume_ml(label_mask.astype(np.uint8), voxel_volume)

#         # keep same lesion-size rule as your radiomics pipeline
#         if volume_ml < 1.0:
#             continue

#         lesion_values = img_array[label_mask]
#         if lesion_values.size == 0:
#             continue

#         lesion_suvmax = float(np.max(lesion_values))
#         lesion_mtvs.append(float(volume_ml))

#         if suvmax_patient is None:
#             suvmax_patient = lesion_suvmax
#         else:
#             suvmax_patient = max(suvmax_patient, lesion_suvmax)

#     if len(lesion_mtvs) == 0:
#         return None

#     tmtv = float(np.sum(lesion_mtvs))

#     # In this lesion-based setup, patient-level MTV and TMTV are identical
#     return {
#         "SUVmax": float(suvmax_patient),
#         "MTV": tmtv,
#         "TMTV": tmtv,
#         "num_lesions_used": int(len(lesion_mtvs))
#     }


def compute_patient_pet_biomarkers(img_array, seg_array, voxel_volume):
    """
    Patient-level PET biomarkers

    SUVmax = maximum SUV across all lesions
    MTV    = largest lesion MTV (mL)
    TMTV   = sum of MTVs of all lesions (mL)

    Assumes each non-zero label in seg_array represents a separate lesion.
    """

    unique_labels = np.unique(seg_array)

    suvmax_patient = None
    lesion_mtvs = []

    for label in unique_labels:
        if label == 0:
            continue

        label_mask = (seg_array == label)

        volume_ml = get_segmentation_volume_ml(
            label_mask.astype(np.uint8),
            voxel_volume
        )

        # keep same lesion-size rule as your radiomics pipeline
        if volume_ml < 1.0:
            continue

        lesion_values = img_array[label_mask]

        if lesion_values.size == 0:
            continue

        lesion_suvmax = float(np.max(lesion_values))

        lesion_mtvs.append(float(volume_ml))

        if suvmax_patient is None:
            suvmax_patient = lesion_suvmax
        else:
            suvmax_patient = max(suvmax_patient, lesion_suvmax)

    if len(lesion_mtvs) == 0:
        return None

    # Total MTV across all lesions
    tmtv = float(np.sum(lesion_mtvs))

    # MTV of the largest lesion
    mtv = float(np.max(lesion_mtvs))

    return {
        "SUVmax": float(suvmax_patient),
        "MTV": mtv,
        "TMTV": tmtv,
        "num_lesions_used": int(len(lesion_mtvs))
    }


def summarize_series(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return "NA"
    return f"{x.median():.2f} [{x.quantile(0.25):.2f}–{x.quantile(0.75):.2f}]"


def build_table1_biomarkers(root_dir):
    data_directories_dir = os.path.join(root_dir, "dataset_directories.json")
    dict_dataset = read_json(data_directories_dir)

    center_info = {
        "Masih-SUV": os.path.join(root_dir, "Masih-SUV"),
        "Razavi-SUV": os.path.join(root_dir, "Razavi-SUV"),
    }

    rows = []

    for center_name, center_root in center_info.items():
        dataset_dicts = dict_dataset.get(center_name, [])

        for item in tqdm(dataset_dicts, desc=f"Processing {center_name}"):
            sample_dir_name, sample_info = list(item.items())[0]

            img_name = sample_info["image"]
            seg_name = sample_info["segmentation"]

            img_dir = os.path.join(center_root, sample_dir_name, img_name.replace(".nii", "_prep_img.nii.gz"))
            seg_dir = os.path.join(center_root, sample_dir_name, seg_name.replace(".nrrd", "_prep_seg.nii.gz"))

            if not os.path.exists(img_dir) or not os.path.exists(seg_dir):
                print(f"Missing file for {center_name}/{sample_dir_name}, skipping.")
                continue

            img_sitk = read_sitk_image(img_dir)
            seg_sitk = read_sitk_image(seg_dir)

            img_array = sitk_image_to_array(img_sitk)
            seg_array = sitk_image_to_array(seg_sitk)

            voxel_volume = get_voxel_volume(seg_sitk)

            biomarkers = compute_patient_pet_biomarkers(img_array, seg_array, voxel_volume)
            if biomarkers is None:
                continue

            label = sample_info.get("Binary Label", None)
            if label not in ["HL", "NHL"]:
                print(f"Unknown label for {center_name}/{sample_dir_name}: {label}")
                continue

            rows.append({
                "center": center_name,
                "patient_id": sample_dir_name,
                "Binary Label": label,
                **biomarkers
            })

    df = pd.DataFrame(rows)

    # -------- Table 1 style summary --------
    summary_rows = []
    for center in ["Masih-SUV", "Razavi-SUV"]:
        for label in ["HL", "NHL"]:
            sub = df[(df["center"] == center) & (df["Binary Label"] == label)]

            summary_rows.append({
                "Center": center,
                "Label": label,
                "N": len(sub),
                "SUVmax, median [IQR]": summarize_series(sub["SUVmax"]),
                "MTV (mL), median [IQR]": summarize_series(sub["MTV"]),
                "TMTV (mL), median [IQR]": summarize_series(sub["TMTV"]),
            })

    summary_df = pd.DataFrame(summary_rows)

    print("\nPatient-level biomarker summary:")
    print(summary_df.to_string(index=False))

    return df, summary_df


if __name__ == "__main__":
    data_root = "../../new_dataset"
    patient_df, summary_df = build_table1_biomarkers(data_root)

    # optional save
    patient_df.to_csv("patient_pet_biomarkers.csv", index=False)
    summary_df.to_csv("table1_pet_biomarkers_summary.csv", index=False)

    '''
    Patient-level biomarker summary:
    Center Label  N SUVmax, median [IQR] MTV (mL), median [IQR] TMTV (mL), median [IQR]
    Masih-SUV    HL 85   11.56 [8.11–14.48]    24.87 [13.96–51.68]    86.67 [30.07–213.70]
    Masih-SUV   NHL 66   18.14 [9.61–25.20]   39.23 [12.54–113.65]    96.71 [26.23–334.76]
    Razavi-SUV    HL 36  14.13 [10.30–16.79]    26.95 [14.63–66.22]    86.28 [39.87–134.01]
    Razavi-SUV   NHL 44  20.33 [13.78–28.11]   69.26 [28.84–165.31]   152.08 [59.15–355.11]
    '''
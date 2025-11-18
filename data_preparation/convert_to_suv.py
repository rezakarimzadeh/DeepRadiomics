import os
import math
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import SimpleITK as sitk
from glob import glob

def load_dicom_series(input_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in: {input_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image, dicom_names

def extract_dicom_metadata(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    weight = float(ds.PatientWeight) * 1000
    injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
    acquisition_time = ds.AcquisitionTime
    injection_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime

    def time_to_sec(t): return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    dt = max(0, time_to_sec(acquisition_time) - time_to_sec(injection_time))
    corrected_dose = injected_dose * math.exp(-math.log(2) * dt / half_life)
    return weight, corrected_dose

def compute_suv(image, weight, dose, factor):
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    suv_array = (image_array * weight) / dose
    suv_array *= factor
    suv_image = sitk.GetImageFromArray(suv_array)
    suv_image.CopyInformation(image)
    return suv_image

def save_suv_image(suv_image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(suv_image, output_path)

def calculate_stats(image):
    array = sitk.GetArrayFromImage(image)
    return np.min(array), np.max(array), np.mean(array)


def convert_to_suv(root):
    patient_names = get_list_of_patients(root)
    suv_stats_list = []
    for patient_id in patient_names:
        print(f"=== Processing {patient_id} ===")
        try:
            dicom_dir = os.path.join(root, patient_id, f"{patient_id} PET")
            dicom_image, dicom_names = load_dicom_series(dicom_dir)
            weight, dose = extract_dicom_metadata(dicom_names[0])

            suv_img = compute_suv(dicom_image, weight, dose, factor=1.0)

            output_nac_path = os.path.join(root, patient_id, f"SUV_PET.nii.gz")
            save_suv_image(suv_img, output_nac_path)
            suv_stats = calculate_stats(suv_img)
            print(f"SUVmin: {suv_stats[0]:.4f}, SUVmax: {suv_stats[1]:.4f}, SUVmean: {suv_stats[2]:.4f}")

            suv_stats_list.append({
                'patient_id': patient_id,
                'mean': suv_stats[2],
                'max': suv_stats[1],
            })

        except Exception as e:
            print(f"Error processing {patient_id}: {e}\n")

def get_list_of_patients(root):
    with os.scandir(root) as it:
        patient_dirs = [e.name for e in it if e.is_dir()]
    return patient_dirs

if __name__ == "__main__":
    convert_to_suv("../../dataset")

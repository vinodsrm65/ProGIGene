import pandas as pd
import boto3
import numpy as np

# Load raw clinical data
df = pd.read_csv('clinical_data.csv')

# Rename columns
df = df.rename(columns={
    '_PATIENT': 'patient_id',
    'cancer type abbreviation': 'cancer_type',
    'age_at_initial_pathologic_diagnosis': 'age_at_diagnosis',
    'ajcc_pathologic_tumor_stage': 'tumor_stage',
    'OS': 'os', 'DSS': 'dss', 'DFI': 'dfi', 'PFI': 'pfi',
    'OS.time': 'os_time', 'DSS.time': 'dss_time',
    'DFI.time': 'dfi_time', 'PFI.time': 'pfi_time',
})

# Subset columns
selected_columns = [
    'sample', 'patient_id', 'cancer_type', 'age_at_diagnosis', 'gender',
    'race', 'tumor_stage', 'vital_status', 'tumor_status',
    'last_contact_days_to', 'death_days_to',
    'new_tumor_event_type', 'new_tumor_event_site', 'new_tumor_event_site_other',
    'new_tumor_event_dx_days_to', 'treatment_outcome_first_course',
    'os', 'os_time', 'dss', 'dss_time', 'dfi', 'dfi_time', 'pfi', 'pfi_time','histological_grade',
    'histological_type','residual_tumor','margin_status','clinical_stage'
]
df = df[selected_columns]

# Filter for GI cancers and valid early stages
gi_types = ['COAD','ESCA', 'LIHC', 'PAAD', 'READ', 'STAD']
early_stages = ['Stage I', 'Stage IA', 'Stage IB', 'Stage IC', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage IIC','Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']

df = df[
    df['cancer_type'].isin(gi_types) &
    df['tumor_stage'].isin(early_stages) &
    df['pfi_time'].notna()
]

# Ensure numeric time
df['pfi_time'] = pd.to_numeric(df['pfi_time'], errors='coerce')
df = df[df['pfi_time'].notna()]

# Optional: remove cases with too-short follow-up
df = df[(df['pfi'] == 1) | (df['pfi_time'] > 365)]

# Create label
df['pfi'] = df['pfi'].astype(int)
df['early_progression_label'] = df.apply(lambda row: 1 if row['pfi'] == 1 and row['pfi_time'] <= 365 else 0, axis=1)

# Save
df.to_csv(f's3://{bucket}/users/tyshaikh/progigene/processed_progigene/elastic/early_stage_clinical_labeled_dataset.csv', index=False)

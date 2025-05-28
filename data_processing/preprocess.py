import pandas as pd
import boto3

# Initialize S3 client
s3 = boto3.client('s3')

# S3 bucket and file information
bucket = 'bucket-name'
key = 'raw_data/clinical_data.csv'

all_clinical_data = pd.read_csv(f's3://{bucket}/{key}', sep=',')

all_clinical_data = all_clinical_data.rename(columns={
    '_PATIENT': 'patient_id',
    'cancer type abbreviation': 'cancer_type',
    'age_at_initial_pathologic_diagnosis': 'age_at_diagnosis',

    'ajcc_pathologic_tumor_stage': 'tumor_stage',
    'OS': 'os',
    'DSS': 'dss',
    'DFI': 'dfi',
    'PFI': 'pfi',
    'OS.time': 'os_time',
    'DSS.time': 'dss_time',
    'DFI.time': 'dfi_time',
    'PFI.time': 'pfi_time',
})

selected_columns = ['sample',
                    'patient_id', 
                    'cancer_type',
                    'age_at_diagnosis', 
                    'gender',
                    'race',
                    'tumor_stage',
                    'vital_status',
                    'tumor_status',
                    'last_contact_days_to',
                    'death_days_to',
                    'new_tumor_event_type',
                    'new_tumor_event_site',
                    'new_tumor_event_site_other',
                    'new_tumor_event_dx_days_to',
                    'treatment_outcome_first_course',
                    'os',
                    'os_time',
                    'dss',
                    'dss_time',
                    'dfi',
                    'dfi_time',
                    'pfi',
                    'pfi_time'
                    ]
all_clinical_data = all_clinical_data[selected_columns]

gi_cancer_types = ['COAD',
                   'ESCA',
                   'LIHC',
                   'PAAD',
                   'READ',
                   'STAD']
gi_clinical_data = all_clinical_data[all_clinical_data['cancer_type'].isin(gi_cancer_types)]
early_stage_clinical_labeled_dataset = gi_clinical_data[
    gi_clinical_data['tumor_stage'].str.contains("Stage I|Stage II|Stage III", case=False, na=False) &
    gi_clinical_data['pfi_time'].notna()
].copy()

early_stage_clinical_labeled_dataset['early_progression_label'] = early_stage_clinical.apply(
    lambda row: 1 if row['pfi'] == 1 and row['pfi_time'] <= 365 else 0,
    axis=1
)

early_stage_clinical_labeled_dataset.to_csv(f's3://{bucket}/processed_progigene/early_stage_clinical_labeled_dataset.csv', index=False)

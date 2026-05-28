import csv
from pathlib import Path
from typing import List, Dict, Any
import datetime

def load_mimic_dataset(data_dir: str | Path, metadata_csv: str | Path = None) -> List[Dict[str, Any]]:
    """
    Load MIMIC-CXR dataset from a local directory.
    Returns a list of dicts with keys: 
    [patient_id, study_id, image_path, report_text, study_date]
    """
    data_dir = Path(data_dir)
    records = []
    
    # If a metadata CSV is provided, use it to accurately map patient/study and dates.
    if metadata_csv and Path(metadata_csv).exists():
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('subject_id', row.get('patient_id'))
                study_id = row.get('study_id')
                study_date = row.get('StudyDate', row.get('study_date'))
                
                # Format date to YYYY-MM-DD if it comes as YYYYMMDD
                if study_date and len(study_date) == 8 and study_date.isdigit():
                    study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
                
                # Construct expected paths based on typical MIMIC-CXR structure
                # e.g., p10/p10000032/s50414267/
                study_path = data_dir / f"p{str(patient_id)[:2]}" / f"p{patient_id}" / f"s{study_id}"
                
                # Fallback path if data is organized flat: data_dir/patient_id/study_id
                alt_study_path = data_dir / str(patient_id) / str(study_id)
                
                active_path = study_path if study_path.exists() else alt_study_path
                
                image_files = list(active_path.glob("*.jpg"))
                report_files = list(active_path.glob("*.txt"))
                
                if image_files and report_files:
                    image_path = str(image_files[0]) # take the first image if multiple
                    with open(report_files[0], 'r', encoding='utf-8') as rf:
                        report_text = rf.read()
                        
                    records.append({
                        "patient_id": str(patient_id),
                        "study_id": str(study_id),
                        "image_path": str(image_path),
                        "report_text": report_text,
                        "study_date": study_date
                    })
    else:
        # Fallback: scan directory structure directly if no metadata CSV is provided.
        # Expects structure like data_dir/patient_id/study_id/
        if data_dir.exists():
            for patient_dir in data_dir.iterdir():
                if not patient_dir.is_dir(): continue
                
                for study_dir in patient_dir.iterdir():
                    if not study_dir.is_dir(): continue
                    
                    image_files = list(study_dir.glob("*.jpg"))
                    report_files = list(study_dir.glob("*.txt"))
                    
                    if image_files and report_files:
                        image_path = str(image_files[0])
                        report_file = report_files[0]
                        
                        with open(report_file, 'r', encoding='utf-8') as rf:
                            report_text = rf.read()
                            
                        # Extract a fallback date from file modification time
                        mod_time = report_file.stat().st_mtime
                        study_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                        
                        records.append({
                            "patient_id": patient_dir.name,
                            "study_id": study_dir.name,
                            "image_path": image_path,
                            "report_text": report_text,
                            "study_date": study_date
                        })
                        
    return records

def load_patient_history(records: List[Dict[str, Any]], patient_id: str) -> List[Dict[str, Any]]:
    """
    Returns all records for a patient sorted by study_date ascending.
    """
    patient_records = [r for r in records if r["patient_id"] == str(patient_id)]
    return sorted(patient_records, key=lambda x: x["study_date"])

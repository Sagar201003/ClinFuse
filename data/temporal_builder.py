from typing import List, Dict, Any

def build_patient_timelines(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Builds a per-patient timeline dict from a list of records.
    Sorts each patient's records chronologically by study_date.
    
    Format:
    {patient_id: [{study_date, study_id, image_path, report_text}, ...]}
    """
    timelines: Dict[str, List[Dict[str, Any]]] = {}
    
    for record in records:
        patient_id = record["patient_id"]
        
        timeline_entry = {
            "study_date": record["study_date"],
            "study_id": record["study_id"],
            "image_path": record["image_path"],
            "report_text": record["report_text"]
        }
        
        if patient_id not in timelines:
            timelines[patient_id] = []
            
        timelines[patient_id].append(timeline_entry)
        
    # Sort chronologically for each patient
    for patient_id in timelines:
        timelines[patient_id] = sorted(timelines[patient_id], key=lambda x: x["study_date"])
        
    return timelines

def get_temporal_context(
    timelines: Dict[str, List[Dict[str, Any]]], 
    patient_id: str, 
    current_study_id: str
) -> List[Dict[str, Any]]:
    """
    Returns all records BEFORE the current study for a given patient.
    Assumes the timeline for the patient is already sorted chronologically.
    """
    if patient_id not in timelines:
        return []
        
    patient_records = timelines[patient_id]
    context_records = []
    
    # Since records are sorted chronologically, we iterate until we hit the current study
    for record in patient_records:
        if record["study_id"] == str(current_study_id):
            break
        context_records.append(record)
        
    return context_records

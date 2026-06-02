import pandas as pd
import os
import random
import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image

def unpack_hf_dataset(parquet_path: str, output_dir: str, num_samples: int = 100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Parquet file from {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return
        
    print(f"Loaded {len(df)} total records. Extracting {num_samples} samples...")
    
    metadata_rows = []
    
    # Slice to extract only a subset of samples
    df_subset = df.head(num_samples)
    
    for idx, row in df_subset.iterrows():
        # 1. Generate consistent synthetic IDs
        patient_id = f"100000{idx:02d}" # format: p10000000
        study_id = f"50414{idx:03d}"   # format: s50414000
        
        # 2. Generate a random chronological date between 2010 and 2020
        # This gives our Temporal Retrieval module real historical context to filter on!
        start_date = datetime.date(2010, 1, 1)
        random_days = random.randint(0, 3650)
        study_date = start_date + datetime.timedelta(days=random_days)
        
        # 3. Build the exact PhysioNet directory structure
        # MIMIC format: data_dir/p10/p10000032/s50414267/
        p_prefix = f"p{patient_id[:2]}"
        study_dir = output_dir / p_prefix / f"p{patient_id}" / f"s{study_id}"
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Extract and save the Image
        image_data = row.get("image")
        if image_data:
            try:
                # HF stores image bytes either in a dict {'bytes': b'...'} or directly as bytes
                img_bytes = image_data.get('bytes') if isinstance(image_data, dict) else image_data
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                image_path = study_dir / f"{study_id}.jpg"
                image.save(image_path)
            except Exception as e:
                print(f"Failed to extract image for index {idx}: {e}")
                
        # 5. Extract and save the Radiology Report Text
        findings = row.get("findings", "")
        impression = row.get("impression", "")
        if pd.isna(findings): findings = ""
        if pd.isna(impression): impression = ""
        
        report_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
        report_path = study_dir / f"{study_id}.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        # 6. Add to Metadata CSV mapping
        metadata_rows.append({
            "subject_id": patient_id,
            "study_id": study_id,
            "StudyDate": study_date.strftime("%Y%m%d") # mimic raw date format YYYYMMDD
        })
        
    # Write the master metadata CSV
    meta_df = pd.DataFrame(metadata_rows)
    csv_path = output_dir / "mimic-cxr-2.0.0-metadata.csv"
    meta_df.to_csv(csv_path, index=False)
    
    print(f"Done! Successfully extracted {num_samples} records to '{output_dir}'")
    print(f"Metadata CSV saved to: {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default="DatasetHF/data/train-00000-of-00002.parquet")
    parser.add_argument("--output", type=str, default="data/mimic-cxr-mock")
    parser.add_argument("--samples", type=int, default=100) # Unpack 100 images by default
    args = parser.parse_args()
    
    unpack_hf_dataset(args.parquet, args.output, args.samples)

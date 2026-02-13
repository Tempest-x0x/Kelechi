#!/usr/bin/env python3
"""
SIGHT Data Pipeline Runner

Run this script to:
1. Extract ZIP files from historical_data/
2. Convert CSV data to Parquet format
3. Resample to 15m and 1H timeframes
4. Validate output data quality

Usage:
    python -m sight.run_pipeline
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sight.data.pipeline import DataPipeline, DataValidator


def main():
    print("=" * 60)
    print("SIGHT DATA PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline(
        historical_data_dir="historical_data",
        output_dir="data/parquet",
        n_workers=4
    )
    
    # Run pipeline
    print("\n[1/3] Running data extraction and conversion...")
    results = pipeline.run_pipeline()
    
    if not results:
        print("ERROR: No data processed!")
        return 1
    
    # Validate output
    print("\n[2/3] Validating output...")
    validation = pipeline.validate_output()
    
    valid_count = sum(
        sum(1 for tf, valid in tfs.items() if valid)
        for tfs in validation.values()
    )
    total_count = sum(len(tfs) for tfs in validation.values())
    
    print(f"Validation: {valid_count}/{total_count} files valid")
    
    # Print summary
    print("\n[3/3] Data Summary:")
    summary = pipeline.get_data_summary()
    
    if len(summary) > 0:
        print(summary.to_string())
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

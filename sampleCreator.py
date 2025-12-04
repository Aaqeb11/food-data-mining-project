#!/usr/bin/env python3
"""
file_column_extractor.py
Extract column names from multiple Excel AND CSV files and create sample files.

Handles both Excel (.xlsx, .xls, etc.) and CSV files.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

# === CONFIGURATION ===
INPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/input"
OUTPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/sample"
SAMPLE_ROWS = 5                              # Number of sample rows to extract
EXCEL_EXTENSIONS = ['.xlsx', '.xls', '.xlsm', '.xlsb']
CSV_EXTENSIONS = ['.csv', '.tsv', '.txt']

# === END CONFIGURATION ===


def get_files_by_type(directory):
    """Get all Excel and CSV files from directory, separated by type."""
    excel_files = []
    csv_files = []
    
    for ext in EXCEL_EXTENSIONS:
        excel_files.extend(Path(directory).glob(f'*{ext}'))
    
    for ext in CSV_EXTENSIONS:
        csv_files.extend(Path(directory).glob(f'*{ext}'))
    
    return sorted(excel_files), sorted(csv_files)


def extract_columns_from_excel(filepath):
    """Extract column names from all sheets in an Excel file."""
    print(f"\nProcessing Excel: {filepath.name}")
    
    try:
        excel_file = pd.ExcelFile(filepath)
        sheet_columns = {}
        
        for sheet_name in excel_file.sheet_names:
            print(f"  Sheet: {sheet_name}")
            
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=0)
                columns = df.columns.tolist()
                sheet_columns[sheet_name] = columns
                print(f"    Columns found: {len(columns)}")
                
            except Exception as e:
                print(f"    ERROR reading sheet '{sheet_name}': {e}")
                sheet_columns[sheet_name] = []
        
        return sheet_columns, 'excel'
        
    except Exception as e:
        print(f"  ERROR: Could not process file: {e}")
        return {}, 'excel'


def extract_columns_from_csv(filepath):
    """Extract column names from a CSV file."""
    print(f"\nProcessing CSV: {filepath.name}")
    
    try:
        # Try reading with different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, nrows=0, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"  ERROR: Could not decode file with any encoding")
            return {}, 'csv'
        
        columns = df.columns.tolist()
        print(f"  Columns found: {len(columns)}")
        
        # CSV files are treated as single "sheet"
        sheet_columns = {'Data': columns}
        return sheet_columns, 'csv'
        
    except Exception as e:
        print(f"  ERROR: Could not process file: {e}")
        return {}, 'csv'


def create_sample_excel(filepath, output_dir, sample_rows=5):
    """Create sample Excel file with first N rows from each sheet."""
    try:
        excel_file = pd.ExcelFile(filepath)
        
        base_name = filepath.stem
        sample_filename = f"{base_name}_SAMPLE_{sample_rows}rows.xlsx"
        sample_path = os.path.join(output_dir, sample_filename)
        
        with pd.ExcelWriter(sample_path, engine='openpyxl') as writer:
            for sheet_name in excel_file.sheet_names:
                try:
                    df_sample = pd.read_excel(
                        filepath, 
                        sheet_name=sheet_name, 
                        nrows=sample_rows
                    )
                    
                    df_sample.to_excel(
                        writer, 
                        sheet_name=sheet_name, 
                        index=False
                    )
                    
                except Exception as e:
                    print(f"    Warning: Could not sample sheet '{sheet_name}': {e}")
        
        print(f"  ✓ Sample file created: {sample_filename}")
        return sample_path
        
    except Exception as e:
        print(f"  ERROR creating sample file: {e}")
        return None


def create_sample_csv(filepath, output_dir, sample_rows=5):
    """Create sample CSV file with first N rows."""
    try:
        base_name = filepath.stem
        sample_filename = f"{base_name}_SAMPLE_{sample_rows}rows.csv"
        sample_path = os.path.join(output_dir, sample_filename)
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df_sample = None
        
        for encoding in encodings:
            try:
                df_sample = pd.read_csv(filepath, nrows=sample_rows, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        
        if df_sample is None:
            print(f"  ERROR: Could not decode file")
            return None
        
        df_sample.to_csv(sample_path, index=False, encoding='utf-8')
        
        print(f"  ✓ Sample file created: {sample_filename}")
        return sample_path
        
    except Exception as e:
        print(f"  ERROR creating sample file: {e}")
        return None


def create_column_summary_excel(all_columns_data, output_path):
    """Create Excel summary with columns from all files."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        all_rows = []
        
        for file_info in all_columns_data:
            filename = file_info['filename']
            file_type = file_info['type']
            
            for sheet_name, columns in file_info['sheets'].items():
                for idx, col in enumerate(columns, 1):
                    all_rows.append({
                        'File_Name': filename,
                        'File_Type': file_type.upper(),
                        'Sheet_Name': sheet_name,
                        'Column_Index': idx,
                        'Column_Name': col
                    })
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_excel(writer, sheet_name='All_Columns', index=False)
            
            # Also create a summary by file
            summary_rows = []
            for file_info in all_columns_data:
                total_cols = sum(len(cols) for cols in file_info['sheets'].values())
                summary_rows.append({
                    'File_Name': file_info['filename'],
                    'File_Type': file_info['type'].upper(),
                    'Number_of_Sheets': len(file_info['sheets']),
                    'Total_Columns': total_cols
                })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='File_Summary', index=False)
    
    print(f"\n✓ Column summary Excel created: {output_path}")


def create_column_summary_csv(all_columns_data, output_path):
    """Create CSV summary with all columns from all files."""
    rows = []
    
    for file_info in all_columns_data:
        filename = file_info['filename']
        file_type = file_info['type']
        
        for sheet_name, columns in file_info['sheets'].items():
            for idx, col in enumerate(columns, 1):
                rows.append({
                    'File_Name': filename,
                    'File_Type': file_type.upper(),
                    'Sheet_Name': sheet_name,
                    'Column_Index': idx,
                    'Column_Name': col
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Column summary CSV created: {output_path}")


def create_column_summary_json(all_columns_data, output_path):
    """Create JSON summary with all columns."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_columns_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Column summary JSON created: {output_path}")


def create_text_report(all_columns_data, stats, output_path):
    """Create human-readable text report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FILE COLUMN EXTRACTION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {stats['total_files']}\n")
        f.write(f"  - Excel files: {stats['excel_files']}\n")
        f.write(f"  - CSV files: {stats['csv_files']}\n")
        f.write(f"Total sheets/tables: {stats['total_sheets']}\n")
        f.write(f"Total unique columns: {stats['total_unique_columns']}\n")
        f.write("\n\n")
        
        for file_info in all_columns_data:
            filename = file_info['filename']
            file_type = file_info['type']
            sheets_data = file_info['sheets']
            
            f.write("=" * 80 + "\n")
            f.write(f"FILE: {filename} ({file_type.upper()})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Number of sheets/tables: {len(sheets_data)}\n\n")
            
            for sheet_name, columns in sheets_data.items():
                f.write(f"\n{sheet_name}:\n")
                f.write(f"Columns ({len(columns)}):\n")
                f.write("-" * 80 + "\n")
                
                for idx, col in enumerate(columns, 1):
                    f.write(f"  {idx:3d}. {col}\n")
                
                f.write("\n")
            
            f.write("\n\n")
    
    print(f"✓ Text report created: {output_path}")


def main():
    print("=" * 80)
    print("FILE COLUMN EXTRACTOR & SAMPLE CREATOR")
    print("Supports: Excel (.xlsx, .xls, .xlsm, .xlsb) and CSV files")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample rows: {SAMPLE_ROWS}")
    print()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples_dir = os.path.join(OUTPUT_DIR, "samples")
    reports_dir = os.path.join(OUTPUT_DIR, "column_reports")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Get all files
    excel_files, csv_files = get_files_by_type(INPUT_DIR)
    
    total_files = len(excel_files) + len(csv_files)
    
    if total_files == 0:
        print(f"ERROR: No Excel or CSV files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(excel_files)} Excel file(s) and {len(csv_files)} CSV file(s)")
    print(f"Total: {total_files} files\n")
    
    # Process all files
    all_columns_data = []
    all_unique_columns = set()
    total_sheets = 0
    
    # Process Excel files
    for filepath in excel_files:
        print("\n" + "=" * 80)
        sheet_columns, file_type = extract_columns_from_excel(filepath)
        
        if sheet_columns:
            all_columns_data.append({
                'filename': filepath.name,
                'type': file_type,
                'sheets': sheet_columns
            })
            total_sheets += len(sheet_columns)
            
            for columns in sheet_columns.values():
                all_unique_columns.update(columns)
            
            print(f"\n  Creating sample file...")
            create_sample_excel(filepath, samples_dir, SAMPLE_ROWS)
    
    # Process CSV files
    for filepath in csv_files:
        print("\n" + "=" * 80)
        sheet_columns, file_type = extract_columns_from_csv(filepath)
        
        if sheet_columns:
            all_columns_data.append({
                'filename': filepath.name,
                'type': file_type,
                'sheets': sheet_columns
            })
            total_sheets += len(sheet_columns)
            
            for columns in sheet_columns.values():
                all_unique_columns.update(columns)
            
            print(f"\n  Creating sample file...")
            create_sample_csv(filepath, samples_dir, SAMPLE_ROWS)
    
    # Calculate statistics
    stats = {
        'total_files': len(all_columns_data),
        'excel_files': len(excel_files),
        'csv_files': len(csv_files),
        'total_sheets': total_sheets,
        'total_unique_columns': len(all_unique_columns)
    }
    
    # Create summary reports
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Excel summary
    excel_summary_path = os.path.join(
        reports_dir, 
        f"columns_summary_{timestamp}.xlsx"
    )
    create_column_summary_excel(all_columns_data, excel_summary_path)
    
    # CSV summary
    csv_summary_path = os.path.join(
        reports_dir, 
        f"columns_summary_{timestamp}.csv"
    )
    create_column_summary_csv(all_columns_data, csv_summary_path)
    
    # JSON summary
    json_summary_path = os.path.join(
        reports_dir, 
        f"columns_summary_{timestamp}.json"
    )
    create_column_summary_json(all_columns_data, json_summary_path)
    
    # Text report
    text_report_path = os.path.join(
        reports_dir, 
        f"columns_report_{timestamp}.txt"
    )
    create_text_report(all_columns_data, stats, text_report_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed: {stats['total_files']}")
    print(f"  - Excel files: {stats['excel_files']}")
    print(f"  - CSV files: {stats['csv_files']}")
    print(f"Total sheets/tables: {stats['total_sheets']}")
    print(f"Unique columns across all files: {stats['total_unique_columns']}")
    print(f"\nOutputs:")
    print(f"  - Sample files: {samples_dir}")
    print(f"  - Column reports: {reports_dir}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
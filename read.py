import pandas as pd
import json

def excel_to_json(excel_path, json_path):
    """
    Convert an Excel file with 'input' and 'response' columns to a JSON file.
    
    Args:
        excel_path (str): Path to the input Excel file
        json_path (str): Path where the JSON file should be saved
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Convert to list of dictionaries
    data = df.to_dict(orient='records')
    
    # Save to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage
excel_to_json('datax.xlsx', 'output.json')
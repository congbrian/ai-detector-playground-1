import csv
from pathlib import Path

def extract_small_csv(input_file: str, output_file: str, rows_to_extract: int):
    """
    Extract a specified number of rows from a large CSV file and save to a new, smaller CSV file.

    Args:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to the output CSV file.
    rows_to_extract (int): Number of rows to extract (excluding header).

    Returns:
    int: Actual number of rows written (excluding header).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"The file {input_path} does not exist.")

    rows_written = 0

    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        header = next(reader)
        writer.writerow(header)

        # Write the specified number of rows
        for row in reader:
            if rows_written >= rows_to_extract:
                break
            writer.writerow(row)
            rows_written += 1

    print(f"Extracted {rows_written} rows from {input_file} to {output_file}")
    return rows_written

# Usage
input_file = "train.csv"
output_file = "train_smallester.csv"
rows_to_extract = 1000  # Adjust this based on your needs

rows_extracted = extract_small_csv(input_file, output_file, rows_to_extract)
print(f"Total rows extracted: {rows_extracted}")
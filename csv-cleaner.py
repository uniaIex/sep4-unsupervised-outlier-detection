import pandas as pd
import csv
import sys
import re

def clean_csv(input_file, output_file):
    """
    Clean a problematic CSV file for more reliable parsing.
    
    This function:
    1. Identifies the correct number of columns from the header
    2. Properly handles quoting and escaping
    3. Normalizes line endings
    4. Removes or repairs problematic rows
    """
    print(f"Cleaning CSV file: {input_file}")
    
    # First pass: determine the expected number of columns from the header
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        # Read just the header line
        header_line = f.readline().strip()
        # Count the number of fields in the header
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(header_line)
            expected_columns = len(header_line.split(dialect.delimiter))
        except:
            # If sniffer fails, fall back to comma as delimiter
            expected_columns = len(header_line.split(','))
    
    print(f"Expected number of columns: {expected_columns}")
    
    # Second pass: process the entire file line by line
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # Read the first line (header) and write it unchanged
        header = infile.readline().strip()
        outfile.write(header + '\n')
        
        # Process the rest of the file
        line_number = 1
        fixed_count = 0
        skipped_count = 0
        
        for line in infile:
            line_number += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                skipped_count += 1
                continue
            
            # Clean the line
            # 1. Handle quotes within fields
            cleaned_line = fix_quotes_in_line(line)
            
            # 2. Count fields after cleaning
            fields = count_fields(cleaned_line)
            
            # 3. Handle cases with too many or too few fields
            if fields != expected_columns:
                if fields > expected_columns:
                    # Too many fields - try to fix by combining quoted fields
                    cleaned_line = fix_excess_fields(cleaned_line, expected_columns)
                    fields = count_fields(cleaned_line)
                elif fields < expected_columns:
                    # Too few fields - add empty fields at the end
                    cleaned_line = cleaned_line + ',' * (expected_columns - fields)
                    fields = count_fields(cleaned_line)
            
            # Check if the line is now valid
            if fields == expected_columns:
                outfile.write(cleaned_line + '\n')
                fixed_count += 1
            else:
                # If still invalid, skip it
                skipped_count += 1
                print(f"Skipping line {line_number}: Found {fields} fields instead of {expected_columns}")
    
    print(f"Cleaning completed: {fixed_count} lines fixed, {skipped_count} lines skipped")
    print(f"Cleaned CSV saved to: {output_file}")

def fix_quotes_in_line(line):
    """Fix inconsistent quoting in a CSV line"""
    # Normalize all quotes to double quotes
    line = line.replace("'", '"')
    
    # Find fields that should be quoted but aren't
    # This is a simplified approach - a more advanced approach would use regex
    in_quotes = False
    chars = list(line)
    
    for i in range(len(chars)):
        if chars[i] == '"':
            in_quotes = not in_quotes
        elif chars[i] == ',' and in_quotes:
            # If we're inside quotes and find a comma, it's part of the field
            pass
    
    # If we end with quotes still open, close them
    if in_quotes:
        line += '"'
    
    return line

def count_fields(line):
    """Count the number of fields in a CSV line, accounting for quoted commas"""
    # Basic CSV parser to count fields correctly
    fields = 0
    in_quotes = False
    i = 0
    
    while i < len(line):
        if line[i] == '"':
            # Toggle quote status
            in_quotes = not in_quotes
            i += 1
        elif line[i] == ',' and not in_quotes:
            # Only count commas outside of quotes as field separators
            fields += 1
            i += 1
        else:
            i += 1
    
    # Count the last field
    fields += 1
    return fields

def fix_excess_fields(line, expected_columns):
    """Try to fix a line with too many fields"""
    fields = []
    current_field = ""
    in_quotes = False
    
    # Parse the line into fields
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
            current_field += char
        elif char == ',' and not in_quotes:
            fields.append(current_field)
            current_field = ""
        else:
            current_field += char
    
    # Add the last field
    fields.append(current_field)
    
    # If we have too many fields, combine excess fields
    if len(fields) > expected_columns:
        # Combine excess fields into the last expected field
        combined = fields[:expected_columns-1]
        combined.append(','.join(fields[expected_columns-1:]))
        return ','.join(combined)
    
    return line

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python csv_cleaner.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    clean_csv(input_file, output_file)
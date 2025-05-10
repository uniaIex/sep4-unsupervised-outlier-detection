import csv
import re

# Map known corrupted sequences to correct letters (based on your example)
CORRUPTION_MAP = {
    r"\+//3//Q-": "á",
    r"\+//3//Q": "á",
    r"\+//3//QA7": "í",
    r"\+AC0": "-",
    r"\+AD0": "=",
    r"\+//f/9//0": "é",
    r"\+//3//f/9//0": "é",
    r"\+//3//A7": "í",
    r"\+//3//A1": "á",
    r"\+//3//A9": "é",
    r"\+//3//BA": "ú",
    r"\+//3//B9": "ù",
    r"\+//3//B6": "ö",
    r"\+//3//F6": "ö",
    r"\+//3//E9": "é",
    r"\+//3//F3": "ó",
    r"\+//3//FA": "ú",
    r"\+//3//E1": "á",
    r"\+//3//ED": "í",
    r"\+//3//C9": "É",
    r"\+//3//D3": "Ó",
    r"\+//3//DA": "Ú",
    r"\+//3//C1": "Á",
    r"\+//3//CD": "Í",
    r"\+//3//9A": "š",
    r"\+//3//9B": "ž",
    r"\+//3//A8": "è",
    r"\+//3//A2": "â",
    r"\+//3//E0": "à",
}

def repair_corrupt_text(text):
    for pattern, replacement in CORRUPTION_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text

def repair_csv(input_path, output_path='repaired_output.csv'):
    with open(input_path, 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        repaired_rows = []
        for row in reader:
            repaired_row = [repair_corrupt_text(cell) for cell in row]
            repaired_rows.append(repaired_row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(repaired_rows)

    print(f"Repaired CSV saved as: {output_path}")

# Example usage:
if __name__ == '__main__':
    repair_csv('data-preview-ted.csv', 'repaired_output.csv')

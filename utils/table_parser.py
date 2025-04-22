# hmafqa/utils/table_parser.py
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union

def parse_html_table(html: str) -> pd.DataFrame:
    """
    Parse an HTML table into a pandas DataFrame.
    
    Args:
        html: HTML string containing a table
        
    Returns:
        Pandas DataFrame
    """
    try:
        # Use pandas to read HTML
        tables = pd.read_html(html)
        if not tables:
            raise ValueError("No tables found in HTML")
        
        # Return the first table
        return tables[0]
    except Exception as e:
        print(f"Error parsing HTML table: {e}")
        # Return empty DataFrame
        return pd.DataFrame()

def extract_table_from_text(text: str) -> pd.DataFrame:
    """
    Extract table structure from text and convert to DataFrame.
    
    Args:
        text: Text potentially containing table-like structure
        
    Returns:
        Pandas DataFrame
    """
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return pd.DataFrame()
    
    # Detect table delimiter
    delimiters = [('|', 0), ('\t', 0), (',', 0)]
    
    for line in lines[:5]:  # Check first few lines
        for i, (delimiter, count) in enumerate(delimiters):
            delimiters[i] = (delimiter, count + line.count(delimiter))
    
    # Use the delimiter with the highest count
    delimiter, _ = max(delimiters, key=lambda x: x[1])
    
    if delimiter == '|':
        # Parse markdown-style table
        header_line = -1
        for i, line in enumerate(lines):
            if line.startswith('|') and '---' in line:
                header_line = i - 1
                break
        
        if header_line >= 0:
            # Clean and parse headers
            headers = lines[header_line].strip('|').split('|')
            headers = [h.strip() for h in headers]
            
            # Clean and parse data
            data = []
            for line in lines[header_line + 2:]:
                if '|' in line:
                    row = line.strip('|').split('|')
                    row = [cell.strip() for cell in row]
                    data.append(row)
            
            return pd.DataFrame(data, columns=headers)
    
    # If not a markdown table, try to parse as CSV
    try:
        return pd.DataFrame([line.split(delimiter) for line in lines])
    except:
        pass
    
    # If all else fails, try to detect fixed-width format
    try:
        # Look for consistent spacing patterns
        spaces = []
        for i, line in enumerate(lines[:10]):  # Check first few lines
            pos = [match.start() for match in re.finditer(r'\s{2,}', line)]
            if pos:
                spaces.append(pos)
        
        # If consistent spacing pattern found
        if spaces and all(len(s) == len(spaces[0]) for s in spaces):
            columns = [0] + [pos + 1 for pos in spaces[0]] + [len(lines[0])]
            
            # Extract data using column positions
            data = []
            for line in lines:
                row = []
                for i in range(len(columns) - 1):
                    start = columns[i]
                    end = columns[i + 1] if i + 1 < len(columns) else len(line)
                    row.append(line[start:end].strip())
                data.append(row)
            
            return pd.DataFrame(data)
    except:
        pass
    
    # If everything fails, just return a DataFrame with a single column
    return pd.DataFrame(lines, columns=['text'])

def table_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a markdown table.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Markdown table as string
    """
    if df.empty:
        return ""
    
    # Convert to markdown
    headers = df.columns.tolist()
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_line = "| " + " | ".join(["---" for _ in headers]) + " |"
    
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
        rows.append(row_str)
    
    return header_line + "\n" + separator_line + "\n" + "\n".join(rows)
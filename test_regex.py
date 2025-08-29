import pandas as pd
import re

# Test data
csv_content = (
    "name,email,phone,notes,ip_address,dob\n"
    "Alice Smith,alice@example.com,+1-415-555-0199,Call after 5pm,192.168.1.10,1990-05-15\n"
    "Bob Johnson,bob@acme.co,4155550188,VIP Customer,203.0.113.45,1985/03/22\n"
    "Charlie Brown,charlie+test@gmail.com,555-867-5309,Met at conference,198.51.100.2,12-01-1992\n"
)

# Create DataFrame
df = pd.DataFrame([row.split(',') for row in csv_content.strip().split('\n')])
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

print("DataFrame:")
print(df)
print("\nColumns:", df.columns.tolist())
print("\nData types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# Test regex patterns
pii_regex = {
    "EMAIL": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE": r"(?:(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})",
    "SSN": r"\b\d{3}-?\d{2}-?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "DOB": r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
}

print("\n--- Testing Regex Patterns ---")
for col in df.columns:
    print(f"\nColumn: {col}")
    series = df[col].astype(str).fillna("")
    print(f"Values: {series.tolist()}")
    
    for pii_type, pattern in pii_regex.items():
        # Test individual values
        matches = []
        for i, val in enumerate(series):
            if re.search(pattern, val):
                matches.append(f"row {i}: '{val}'")
        
        if matches:
            print(f"  {pii_type}: {len(matches)} matches - {matches}")
        else:
            print(f"  {pii_type}: No matches")
            
        # Test with pandas str.contains
        try:
            count = series.str.contains(pattern, regex=True).sum()
            print(f"  {pii_type} (pandas): {count} matches")
        except Exception as e:
            print(f"  {pii_type} (pandas): Error - {e}")

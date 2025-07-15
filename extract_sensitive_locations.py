import pandas as pd

# Path to your large CSV file
csv_path = 'Data/COSA_Pavement.csv'  # or 'Data/COSA_Pavement_latlon.csv'
output_path = 'possible_sensitive_locations.csv'

# Keywords to search for
keywords = ['school', 'hospital', 'medical', 'clinic', 'senior', 'elder', 'center', 'community']

# Function to check if any keyword is in a string (case-insensitive)
def contains_keywords(val):
    if pd.isnull(val):
        return False
    val = str(val).lower()
    return any(kw in val for kw in keywords)

# Read in chunks to handle large files
chunk_size = 10000
matches = []

for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str, low_memory=False):
    # Check all string columns for keyword matches
    mask = chunk.applymap(contains_keywords).any(axis=1)
    filtered = chunk[mask]
    if not filtered.empty:
        matches.append(filtered)

if matches:
    result = pd.concat(matches)
    result.to_csv(output_path, index=False)
    print(f"Found {len(result)} possible sensitive locations. Saved to {output_path}")
else:
    print("No possible sensitive locations found.") 
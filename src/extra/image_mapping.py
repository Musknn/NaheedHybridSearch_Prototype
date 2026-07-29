"""
Create image mapping from CSV.
Finds which product_id maps to which image filename.
"""

import pandas as pd
import json
import re
from pathlib import Path
from config import CATALOGUE_PATH

# Load CSV
df = pd.read_csv(CATALOGUE_PATH, low_memory=False)

# Get list of your actual image files
images_dir = Path(__file__).parent.parent / 'frontend' / 'images'
image_files = list(images_dir.glob('*.jpg'))
image_numbers = [f.stem for f in image_files]  # e.g., "27342"

print(f"Found {len(image_files)} image files")
print(f"Example image numbers: {image_numbers[:5]}")

# Try to find where these numbers appear in the CSV
mapping = {}

# Check if image numbers appear in product_id column
product_ids = df['product_id'].astype(str).tolist()
for img_num in image_numbers:
    if img_num in product_ids:
        # Find the SKU for this product_id
        row = df[df['product_id'].astype(str) == img_num]
        if not row.empty:
            sku = str(row.iloc[0]['sku'])
            mapping[sku] = img_num
            print(f"✅ Found: {sku} → {img_num}.jpg (product_id match)")

# If no matches, check if image numbers appear in the image column
if not mapping:
    print("\nChecking image column...")
    for _, row in df.iterrows():
        image_path = str(row.get('image', ''))
        sku = str(row.get('sku', ''))
        if sku and image_path and pd.notna(image_path):
            # Extract number from image path
            match = re.search(r'/(\d+)-\d+\.jpg', image_path)
            if match:
                img_num = match.group(1)
                if img_num in image_numbers:
                    mapping[sku] = img_num
                    print(f"✅ Found: {sku} → {img_num}.jpg")

# Save the mapping
if mapping:
    output_path = Path(__file__).parent.parent / 'frontend' / 'image_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\n✅ Saved {len(mapping)} mappings to {output_path}")
    print("\nSample mappings:")
    for sku, img in list(mapping.items())[:10]:
        print(f"  {sku} → {img}.jpg")
else:
    print("\n❌ No mapping found!")
    print("\nPlease check your CSV for a column that contains these numbers:")
    print(f"  {image_numbers[:10]}")
    print("\nPossible column names: product_id, id, image_id, image, sku")
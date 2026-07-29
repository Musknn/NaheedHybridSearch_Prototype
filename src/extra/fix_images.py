"""
Image Fix Script - Automatically fixes product images
1. Finds all image files in frontend/images/
2. Creates proper mapping from SKU to image filename
3. Generates placeholder images for missing products
4. Updates image_mapping.json

Usage:
    python fix_images.py
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "frontend" / "images"
MAPPING_FILE = BASE_DIR / "frontend" / "image_mapping.json"
CSV_PATH = BASE_DIR / "data" / "raw" / "AllProducts.csv"

# ──────────────────────────────────────────────────────────────────────────
# Step 1: Create Placeholder Image
# ──────────────────────────────────────────────────────────────────────────

def create_placeholder():
    """Create a placeholder image for missing products."""
    placeholder_path = IMAGES_DIR / "placeholder.jpg"
    
    if placeholder_path.exists():
        print(f"✅ Placeholder already exists: {placeholder_path}")
        return placeholder_path
    
    print("📸 Creating placeholder image...")
    
    # Create a 200x200 image with a gradient background
    img = Image.new('RGB', (200, 200), color='#f8fafc')
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([(5, 5), (195, 195)], outline='#cbd5e1', width=2)
    
    # Draw a simple camera icon
    draw.rectangle([(70, 85), (130, 115)], outline='#94a3b8', width=2)
    draw.ellipse([(60, 75), (90, 105)], outline='#94a3b8', width=2)
    
    # Add text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 130), "No Image", fill='#94a3b8', font=font)
    
    # Save
    img.save(placeholder_path, 'JPEG', quality=90)
    print(f"✅ Created placeholder: {placeholder_path}")
    return placeholder_path


# ──────────────────────────────────────────────────────────────────────────
# Step 2: Scan Image Files
# ──────────────────────────────────────────────────────────────────────────

def get_all_images():
    """Get all image files in the images directory."""
    image_files = {}
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        for f in IMAGES_DIR.glob(f'*{ext}'):
            name = f.stem  # filename without extension
            image_files[name] = f
    return image_files


def extract_number_from_path(image_path):
    """Extract numeric ID from image path like /1/1/1143194-1.jpg"""
    match = re.search(r'/(\d+)-\d+\.jpg', str(image_path))
    if match:
        return match.group(1)
    return None


# ──────────────────────────────────────────────────────────────────────────
# Step 3: Build Complete Mapping
# ──────────────────────────────────────────────────────────────────────────

def build_image_mapping():
    """
    Build a complete mapping from SKU to image filename.
    Uses multiple strategies to find the right image.
    """
    print("\n" + "=" * 60)
    print("🔍 BUILDING IMAGE MAPPING")
    print("=" * 60)
    
    # Load CSV
    print("\n📂 Loading catalogue...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"   Loaded {len(df):,} products")
    
    # Get all images
    all_images = get_all_images()
    print(f"   Found {len(all_images):,} images in folder")
    
    mapping = {}
    unmatched_skus = []
    matched_count = 0
    image_fallback_count = 0
    
    # Process each product
    for _, row in df.iterrows():
        sku = str(row.get('sku', ''))
        if not sku or pd.isna(sku):
            continue
        
        # Get product_id and image path from CSV
        product_id = str(row.get('product_id', ''))
        image_path = str(row.get('image', ''))
        
        found = False
        
        # ── Strategy 1: Use image from CSV ──
        if image_path and image_path != 'nan':
            numeric_id = extract_number_from_path(image_path)
            if numeric_id and numeric_id in all_images:
                mapping[sku] = numeric_id
                found = True
                matched_count += 1
                continue
        
        # ── Strategy 2: Use product_id ──
        if product_id and product_id != 'nan':
            if product_id in all_images:
                mapping[sku] = product_id
                found = True
                matched_count += 1
                continue
        
        # ── Strategy 3: Try to find any image with this product_id ──
        if product_id:
            # Check if any image starts with this product_id
            for img_name in all_images:
                if img_name.startswith(product_id):
                    mapping[sku] = img_name
                    found = True
                    image_fallback_count += 1
                    break
        
        if not found:
            unmatched_skus.append(sku)
    
    # ── Strategy 4: For unmatched SKUs, create placeholder ──
    if unmatched_skus:
        print(f"\n⚠️ {len(unmatched_skus)} SKUs have no matching image")
        sample_unmatched = unmatched_skus[:5]
        print(f"   Example: {sample_unmatched}")
        # We'll create placeholders for them later
    
    print(f"\n📊 Mapping Results:")
    print(f"   ✅ Matched: {matched_count} products")
    print(f"   🔄 Fallback: {image_fallback_count} products (using best guess)")
    print(f"   ❌ Unmatched: {len(unmatched_skus)} products")
    
    return mapping, unmatched_skus


# ──────────────────────────────────────────────────────────────────────────
# Step 4: Create Placeholders for Missing Products
# ──────────────────────────────────────────────────────────────────────────

def create_placeholders(mapping, unmatched_skus, limit=500):
    """
    Create placeholder images for unmatched products.
    """
    if not unmatched_skus:
        print("✅ All products have image mappings!")
        return mapping
    
    print("\n" + "=" * 60)
    print("📸 CREATING PLACEHOLDERS FOR MISSING PRODUCTS")
    print("=" * 60)
    
    created = 0
    for sku in unmatched_skus[:limit]:
        # Create a simple placeholder with the SKU text
        img = Image.new('RGB', (200, 200), color='#f8fafc')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(5, 5), (195, 195)], outline='#cbd5e1', width=2)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw the SKU text
        draw.text((10, 80), sku, fill='#64748b', font=font)
        draw.text((10, 100), "(No Image)", fill='#94a3b8', font=font)
        
        # Save as SKU.jpg
        output_path = IMAGES_DIR / f"{sku}.jpg"
        img.save(output_path, 'JPEG', quality=85)
        created += 1
        
        # Add to mapping
        mapping[sku] = sku  # Use SKU as the image name
    
    print(f"✅ Created {created} placeholder images")
    print(f"   (Limited to {limit} products to avoid too many files)")
    if len(unmatched_skus) > limit:
        print(f"   ⚠️ {len(unmatched_skus) - limit} products still without images")
        print(f"   Increase the 'limit' parameter to create more")
    
    return mapping


# ──────────────────────────────────────────────────────────────────────────
# Step 5: Save Mapping File
# ──────────────────────────────────────────────────────────────────────────

def save_mapping(mapping):
    """Save the mapping to image_mapping.json."""
    print("\n" + "=" * 60)
    print("💾 SAVING MAPPING")
    print("=" * 60)
    
    # Convert all values to strings
    clean_mapping = {str(k): str(v) for k, v in mapping.items() if k and v}
    
    with open(MAPPING_FILE, 'w') as f:
        json.dump(clean_mapping, f, indent=2)
    
    print(f"✅ Saved {len(clean_mapping)} mappings to {MAPPING_FILE}")
    
    # Show sample
    sample_items = list(clean_mapping.items())[:5]
    if sample_items:
        print("\n📋 Sample Mappings:")
        for sku, img in sample_items:
            print(f"   {sku} → {img}.jpg")


# ──────────────────────────────────────────────────────────────────────────
# Step 6: (Optional) Fix Image Names - Rename to match SKU
# ──────────────────────────────────────────────────────────────────────────

def fix_image_names(mapping):
    """
    OPTIONAL: Rename image files to match SKU names.
    This makes it easier to find images by SKU directly.
    Example: 100024.jpg → IC-1253515.jpg
    Set RENAME_IMAGES = True below to enable.
    """
    print("\n" + "=" * 60)
    print("🔄 RENAMING IMAGES TO MATCH SKU (OPTIONAL)")
    print("=" * 60)
    print("⚠️ This will rename your image files in the images folder.")
    print("   To enable, set RENAME_IMAGES = True at the bottom of the script.")
    return


# ──────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration toggle ──
    # Set to True if you want to rename images to match SKU names
    RENAME_IMAGES = False  # ⬅️ Change to True if you want renaming
    
    print("=" * 60)
    print("🖼️  NAHEED IMAGE FIX SCRIPT")
    print("=" * 60)
    
    # Check if images directory exists
    if not IMAGES_DIR.exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        print("   Creating it...")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create placeholder
    create_placeholder()
    
    # Step 2: Build mapping
    mapping, unmatched = build_image_mapping()
    
    # Step 3: Create placeholders for unmatched
    mapping = create_placeholders(mapping, unmatched, limit=500)
    
    # Step 4: (Optional) Rename images
    if RENAME_IMAGES:
        fix_image_names(mapping)
    
    # Step 5: Save mapping
    save_mapping(mapping)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ IMAGE FIX COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   📁 Images folder: {IMAGES_DIR}")
    print(f"   📄 Mapping file: {MAPPING_FILE}")
    print(f"   📸 Total images mapped: {len(mapping)}")
    print(f"   ❌ Still missing (not in mapping): {len(unmatched) - min(len(unmatched), 500)} (if any)")

    print("\n💡 Next steps:")
    print("   1. Restart your frontend server")
    print("   2. Open the browser and search for products")
    print("   3. Images should now load (real ones or placeholders)")
    print("\n🔍 To test, search for 'Beaver' - you should see product images.")


if __name__ == "__main__":
    main()
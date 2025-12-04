#!/usr/bin/env python3
"""
fdc_restaurant_foods_preprocess.py
Preprocess USDA FoodData Central for restaurant-ready cooked foods.

Based on actual CSV structure with correct column names.
Filters for restaurant-style prepared foods and creates parquet outputs.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from datetime import datetime

# === CONFIGURATION ===
INPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/input"
OUTPUT_DIR = r"C:/Users/cnaya/Downloads/FoodData_Central_csv_2025-04-24/FoodData_Central_csv_2025-04-24/output"

# Cooking method categories (in priority order)
COOKING_METHODS = {
    'fried': ['fried', 'deep-fried', 'pan-fried', 'deep fried', 'pan fried', 'stir-fried', 'stir fried'],
    'grilled': ['grilled', 'grill', 'barbecued', 'bbq'],
    'baked': ['baked', 'bake'],
    'roasted': ['roasted', 'roast'],
    'boiled': ['boiled', 'boil'],
    'steamed': ['steamed', 'steam'],
    'stewed': ['stewed', 'stew', 'braised', 'braise'],
    'prepared': ['prepared', 'cooked', 'ready-to-eat', 'ready to eat'],
    'sauteed': ['sauteed', 'sautéed', 'sauté', 'saute'],
    'broiled': ['broiled', 'broil'],
    'poached': ['poached', 'poach'],
    'microwaved': ['microwaved', 'microwave'],
    'toasted': ['toasted', 'toast'],
    'raw': ['raw', 'uncooked', 'fresh']
}

# Restaurant/cooked food keywords (for filtering)
COOKED_KEYWORDS = [
    'cooked', 'fried', 'boiled', 'grilled', 'baked', 'roasted', 'stewed',
    'prepared', 'casserole', 'sauteed', 'sautéed', 'broiled', 'pan-fried', 
    'deep-fried', 'poached', 'steamed', 'braised', 'stir-fried', 'simmered',
    'toasted', 'blanched', 'microwaved', 'heated'
]

RESTAURANT_DISH_KEYWORDS = [
    'pizza', 'burger', 'sandwich', 'wrap', 'burrito', 'taco', 'quesadilla',
    'pasta', 'spaghetti', 'lasagna', 'ravioli', 'fettuccine', 'linguine',
    'curry', 'stew', 'soup', 'chili', 'gumbo', 'chowder', 'bisque',
    'salad', 'coleslaw', 'potato salad',
    'fried chicken', 'fried fish', 'chicken nugget', 'chicken tender',
    'french fries', 'fries', 'hash brown', 'tater tot', 'onion ring',
    'pancake', 'waffle', 'french toast', 'omelet', 'omelette', 'scrambled',
    'hot dog', 'sub', 'hoagie', 'panini',
    'rice bowl', 'noodle bowl', 'poke bowl',
    'enchilada', 'chimichanga', 'fajita', 'nachos',
    'pad thai', 'lo mein', 'chow mein', 'fried rice',
    'kebab', 'kabob', 'shawarma', 'gyro',
    'dim sum', 'dumpling', 'spring roll', 'egg roll',
    'tempura', 'teriyaki', 'katsu',
    'meatball', 'meatloaf', 'pot pie', 'quiche',
    'combo', 'meal', 'platter', 'entree', 'entrée'
]

# Exclude raw ingredients and non-restaurant items
EXCLUDE_KEYWORDS = [
    'raw', 'uncooked', 'fresh, raw', 'unprepared',
    'baby food', 'infant formula',
    'supplement', 'vitamin', 'mineral supplement',
    'protein powder', 'shake mix',
    'baking powder', 'baking soda', 'yeast', 'gelatin',
    'oil, pure', 'vinegar, pure', 'salt, table',
    'water, tap', 'water, bottled'
]

# Restaurant chain keywords
RESTAURANT_BRANDS = [
    "mcdonald's", 'mcdonalds', 'burger king', "wendy's", 'wendys',
    'taco bell', 'kfc', 'subway', 'pizza hut', "domino's", 'dominos',
    "papa john's", 'little caesars', "chick-fil-a", 'chipotle', 'panera',
    "applebee's", "chili's", 'olive garden', 'red lobster', 'outback',
    'buffalo wild wings', 'wingstop', "arby's", 'sonic', 'dairy queen',
    "carl's jr", "hardee's", 'jack in the box', 'whataburger', 'five guys',
    'shake shack', 'panda express', "p.f. chang's", 'cheesecake factory'
]

# === END CONFIGURATION ===


def load_csv_safe(filepath, encoding='utf-8'):
    """Load CSV with error handling and multiple encoding attempts."""
    try:
        if os.path.exists(filepath):
            basename = os.path.basename(filepath)
            print(f"Loading: {basename}...", end=' ')
            
            # Try multiple encodings
            encodings = [encoding, 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for enc in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=enc, low_memory=False)
                    print(f"{len(df):,} rows")
                    return df
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print("ERROR: Could not decode file")
                return pd.DataFrame()
        else:
            print(f"WARNING: File not found: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return pd.DataFrame()


def categorize_cooking_method(description):
    """Categorize food by cooking method based on description."""
    if pd.isna(description):
        return 'unknown'
    
    desc_lower = str(description).lower()
    
    # Check each cooking method in priority order
    for method, keywords in COOKING_METHODS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return method
    
    return 'unknown'


def is_restaurant_food(description, data_type, brand_owner=''):
    """Determine if a food item is restaurant-ready."""
    if pd.isna(description):
        return False
    
    desc_lower = str(description).lower()
    dtype_lower = str(data_type).lower()
    brand_lower = str(brand_owner).lower()
    
    # Exclude unwanted items first
    for exclude_word in EXCLUDE_KEYWORDS:
        if exclude_word in desc_lower:
            return False
    
    # Include Survey/FNDDS foods (restaurant-style prepared foods)
    if 'survey' in dtype_lower or 'fndds' in dtype_lower:
        return True
    
    # Check for cooked/prepared keywords
    for keyword in COOKED_KEYWORDS:
        if keyword in desc_lower:
            return True
    
    # Check for restaurant dish keywords
    for keyword in RESTAURANT_DISH_KEYWORDS:
        if keyword in desc_lower:
            return True
    
    # Check for restaurant brands (for branded foods)
    if 'branded' in dtype_lower:
        for brand in RESTAURANT_BRANDS:
            if brand in desc_lower or brand in brand_lower:
                return True
    
    return False


def categorize_nutrient(name):
    """Categorize nutrient by name to standardized categories."""
    if pd.isna(name):
        return None
    
    name_lower = str(name).lower()
    
    # Energy/Calories
    if 'energy' in name_lower or 'calori' in name_lower:
        return 'calories'
    
    # Protein
    if 'protein' in name_lower:
        return 'protein'
    
    # Fat
    if 'total lipid' in name_lower or 'fat, total' in name_lower:
        return 'fat'
    if 'fatty acids, total saturated' in name_lower or 'saturated fat' in name_lower:
        return 'saturated_fat'
    if 'fatty acids, total trans' in name_lower or 'trans fat' in name_lower:
        return 'trans_fat'
    
    # Carbohydrates
    if 'carbohydrate, by difference' in name_lower or 'carbohydrate' in name_lower:
        return 'carbohydrate'
    if 'fiber, total dietary' in name_lower or 'dietary fiber' in name_lower:
        return 'fiber'
    if 'sugars, total' in name_lower or 'total sugars' in name_lower:
        return 'sugar'
    
    # Minerals
    if name_lower.startswith('sodium'):
        return 'sodium'
    if name_lower.startswith('calcium'):
        return 'calcium'
    if name_lower.startswith('iron'):
        return 'iron'
    if name_lower.startswith('potassium'):
        return 'potassium'
    
    # Vitamins
    if 'cholesterol' in name_lower:
        return 'cholesterol'
    if 'vitamin c' in name_lower or 'ascorbic acid' in name_lower:
        return 'vitamin_c'
    if 'vitamin a' in name_lower:
        return 'vitamin_a'
    if 'vitamin d' in name_lower:
        return 'vitamin_d'
    
    return None


def main():
    print("=" * 80)
    print("USDA FoodData Central - Restaurant Foods Preprocessor")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # === STEP 1: Load core datasets ===
    print("STEP 1: Loading core datasets...")
    print("-" * 80)
    
    # Main food table
    food = load_csv_safe(os.path.join(INPUT_DIR, "food.csv"))
    
    # Nutrient data
    food_nutrient = load_csv_safe(os.path.join(INPUT_DIR, "food_nutrient.csv"))
    nutrient = load_csv_safe(os.path.join(INPUT_DIR, "nutrient.csv"))
    
    # Metadata tables
    food_category = load_csv_safe(os.path.join(INPUT_DIR, "food_category.csv"))
    food_portion = load_csv_safe(os.path.join(INPUT_DIR, "food_portion.csv"))
    measure_unit = load_csv_safe(os.path.join(INPUT_DIR, "measure_unit.csv"))
    
    # Survey/FNDDS data
    survey_fndds = load_csv_safe(os.path.join(INPUT_DIR, "survey_fndds_food.csv"))
    wweia_category = load_csv_safe(os.path.join(INPUT_DIR, "wweia_food_category.csv"))
    
    # Branded food data
    branded_food = load_csv_safe(os.path.join(INPUT_DIR, "branded_food.csv"))
    
    if food.empty or food_nutrient.empty or nutrient.empty:
        print("ERROR: Could not load core datasets. Exiting.")
        return
    
    print(f"\nTotal foods in database: {len(food):,}")
    
    # === STEP 1.5: Ensure correct data types ===
    print("\nConverting data types...")
    
    # Food table
    food['fdc_id'] = pd.to_numeric(food['fdc_id'], errors='coerce')
    food['food_category_id'] = pd.to_numeric(food['food_category_id'], errors='coerce')
    
    # Nutrient tables
    food_nutrient['fdc_id'] = pd.to_numeric(food_nutrient['fdc_id'], errors='coerce')
    food_nutrient['nutrient_id'] = pd.to_numeric(food_nutrient['nutrient_id'], errors='coerce')
    food_nutrient['amount'] = pd.to_numeric(food_nutrient['amount'], errors='coerce')
    
    nutrient['id'] = pd.to_numeric(nutrient['id'], errors='coerce')
    
    # Category table
    food_category['id'] = pd.to_numeric(food_category['id'], errors='coerce')
    
    # Portion table
    food_portion['fdc_id'] = pd.to_numeric(food_portion['fdc_id'], errors='coerce')
    food_portion['gram_weight'] = pd.to_numeric(food_portion['gram_weight'], errors='coerce')
    food_portion['measure_unit_id'] = pd.to_numeric(food_portion['measure_unit_id'], errors='coerce')
    
    measure_unit['id'] = pd.to_numeric(measure_unit['id'], errors='coerce')
    
    # Survey tables
    if not survey_fndds.empty:
        survey_fndds['fdc_id'] = pd.to_numeric(survey_fndds['fdc_id'], errors='coerce')
    
    # Branded table
    if not branded_food.empty:
        branded_food['fdc_id'] = pd.to_numeric(branded_food['fdc_id'], errors='coerce')
    
    print("Data types converted")
    print()
    
    # === STEP 2: Merge branded food info ===
    print("STEP 2: Merging branded food information...")
    print("-" * 80)
    
    if not branded_food.empty and 'fdc_id' in branded_food.columns:
        branded_cols = ['fdc_id', 'brand_owner', 'brand_name', 'subbrand_name', 
                       'gtin_upc', 'ingredients', 'branded_food_category']
        branded_cols = [col for col in branded_cols if col in branded_food.columns]
        
        food = food.merge(
            branded_food[branded_cols],
            on='fdc_id',
            how='left'
        )
        print(f"Merged branded food data for {branded_food['fdc_id'].nunique():,} foods")
    print()
    
    # === STEP 3: Filter for restaurant foods ===
    print("STEP 3: Filtering for restaurant-ready foods...")
    print("-" * 80)
    
    # Apply restaurant food filter
    food['brand_owner'] = food.get('brand_owner', '')
    food['is_restaurant'] = food.apply(
        lambda row: is_restaurant_food(
            row['description'], 
            row['data_type'],
            row.get('brand_owner', '')
        ), 
        axis=1
    )
    
    restaurant_foods = food[food['is_restaurant']].copy()
    
    # Add cooking method categorization
    restaurant_foods['cooking_method'] = restaurant_foods['description'].apply(categorize_cooking_method)
    
    print(f"Found {len(restaurant_foods):,} restaurant-ready foods")
    print(f"\nData type breakdown:")
    print(restaurant_foods['data_type'].value_counts().to_string())
    print(f"\nCooking method breakdown:")
    print(restaurant_foods['cooking_method'].value_counts().to_string())
    print()
    
    # === STEP 4: Process nutrient data ===
    print("STEP 4: Processing nutrient data...")
    print("-" * 80)
    
    # Categorize nutrients
    nutrient['nutrient_category'] = nutrient['name'].apply(categorize_nutrient)
    core_nutrients = nutrient[nutrient['nutrient_category'].notna()].copy()
    
    print(f"Core nutrient categories identified: {core_nutrients['nutrient_category'].nunique()}")
    print(f"Categories: {sorted(core_nutrients['nutrient_category'].unique())}")
    print()
    
    # Filter food_nutrient for restaurant foods only
    restaurant_fdc_ids = set(restaurant_foods['fdc_id'].dropna())
    print(f"Filtering {len(food_nutrient):,} nutrient records...")
    
    food_nutrient_filtered = food_nutrient[
        food_nutrient['fdc_id'].isin(restaurant_fdc_ids)
    ].copy()
    
    print(f"Nutrient records for restaurant foods: {len(food_nutrient_filtered):,}")
    
    # Join with nutrient categories
    nutrient_data = food_nutrient_filtered.merge(
        core_nutrients[['id', 'name', 'nutrient_category', 'unit_name']],
        left_on='nutrient_id',
        right_on='id',
        how='inner'
    )
    
    print(f"Core nutrient records: {len(nutrient_data):,}")
    print()
    
    # === STEP 5: Pivot nutrients to wide format ===
    print("STEP 5: Creating wide-format nutrient table...")
    print("-" * 80)
    
    # Pivot to wide format
    nutrient_wide = nutrient_data.pivot_table(
        index='fdc_id',
        columns='nutrient_category',
        values='amount',
        aggfunc='first'
    ).reset_index()
    
    # Get unit information
    nutrient_units = nutrient_data.groupby('nutrient_category')['unit_name'].first().to_dict()
    
    available_nutrients = [col for col in nutrient_wide.columns if col != 'fdc_id']
    print(f"Nutrients available: {len(available_nutrients)}")
    print(f"Columns: {available_nutrients}")
    print()
    
    # === STEP 6: Process portion/serving size data ===
    print("STEP 6: Processing portion data...")
    print("-" * 80)
    
    if not food_portion.empty:
        portions = food_portion[
            food_portion['fdc_id'].isin(restaurant_fdc_ids)
        ].copy()
        
        # Merge with measure units if available
        if not measure_unit.empty and 'measure_unit_id' in portions.columns:
            portions = portions.merge(
                measure_unit[['id', 'name']].rename(columns={'name': 'unit_name'}),
                left_on='measure_unit_id',
                right_on='id',
                how='left'
            )
        
        # Aggregate portion data
        portion_summary = portions.groupby('fdc_id').agg({
            'gram_weight': ['mean', 'min', 'max', 'count'],
            'portion_description': lambda x: ' | '.join(str(v) for v in x.dropna().unique()[:3] if v)
        }).reset_index()
        
        portion_summary.columns = [
            'fdc_id', 'serving_size_g_mean', 'serving_size_g_min',
            'serving_size_g_max', 'portion_count', 'portion_descriptions'
        ]
        
        print(f"Portion data for {len(portion_summary):,} foods")
    else:
        portion_summary = pd.DataFrame(columns=['fdc_id'])
    
    print()
    
    # === STEP 7: Add Survey/FNDDS category info ===
    print("STEP 7: Adding Survey/FNDDS category information...")
    print("-" * 80)
    
    if not survey_fndds.empty:
        # Merge WWEIA categories
        if not wweia_category.empty and 'wweia_category_code' in survey_fndds.columns:
            survey_fndds = survey_fndds.merge(
                wweia_category,
                left_on='wweia_category_code',
                right_on='wweia_food_category',
                how='left'
            )
        
        survey_cols = ['fdc_id', 'food_code', 'wweia_food_category_description']
        survey_cols = [col for col in survey_cols if col in survey_fndds.columns]
        survey_info = survey_fndds[survey_cols]
        
        print(f"Survey/FNDDS data for {len(survey_info):,} foods")
    else:
        survey_info = pd.DataFrame(columns=['fdc_id'])
    
    print()
    
    # === STEP 8: Merge all data ===
    print("STEP 8: Merging all datasets...")
    print("-" * 80)
    
    # Start with food metadata
    metadata_cols = ['fdc_id', 'data_type', 'description', 'food_category_id', 
                    'publication_date', 'cooking_method']
    if 'brand_owner' in restaurant_foods.columns:
        metadata_cols.extend(['brand_owner', 'brand_name', 'branded_food_category'])
    metadata_cols = [col for col in metadata_cols if col in restaurant_foods.columns]
    
    final_df = restaurant_foods[metadata_cols].copy()
    
    # Add category descriptions
    if not food_category.empty:
        food_category_clean = food_category[['id', 'description']].rename(
            columns={'id': 'food_category_id', 'description': 'category_name'}
        )
        final_df = final_df.merge(food_category_clean, on='food_category_id', how='left')
    
    # Merge nutrients
    final_df = final_df.merge(nutrient_wide, on='fdc_id', how='left')
    
    # Merge portions
    if not portion_summary.empty:
        final_df = final_df.merge(portion_summary, on='fdc_id', how='left')
    
    # Merge survey info
    if not survey_info.empty and 'wweia_food_category_description' in survey_info.columns:
        final_df = final_df.merge(
            survey_info[['fdc_id', 'wweia_food_category_description']], 
            on='fdc_id', 
            how='left'
        )
    
    print(f"Final dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
    print()
    
    # === STEP 9: Add computed nutritional metrics ===
    print("STEP 9: Adding computed nutritional metrics...")
    print("-" * 80)
    
    # Calculate calories from macros if missing
    if all(col in final_df.columns for col in ['protein', 'fat', 'carbohydrate']):
        final_df['calculated_calories'] = (
            final_df['protein'].fillna(0) * 4 +
            final_df['fat'].fillna(0) * 9 +
            final_df['carbohydrate'].fillna(0) * 4
        )
        
        # Use reported calories, fall back to calculated
        if 'calories' in final_df.columns:
            final_df['calories_final'] = final_df['calories'].fillna(final_df['calculated_calories'])
        else:
            final_df['calories_final'] = final_df['calculated_calories']
    
    # Macro percentages
    if 'calories_final' in final_df.columns:
        cal_col = final_df['calories_final'].replace(0, np.nan)
        
        if 'protein' in final_df.columns:
            final_df['protein_pct'] = (final_df['protein'].fillna(0) * 4 / cal_col * 100).round(1)
        if 'fat' in final_df.columns:
            final_df['fat_pct'] = (final_df['fat'].fillna(0) * 9 / cal_col * 100).round(1)
        if 'carbohydrate' in final_df.columns:
            final_df['carb_pct'] = (final_df['carbohydrate'].fillna(0) * 4 / cal_col * 100).round(1)
    
    # Nutrient density score (nutrients per 100 calories)
    if 'calories_final' in final_df.columns:
        cal_per_100 = final_df['calories_final'].replace(0, np.nan) / 100
        
        if 'protein' in final_df.columns:
            final_df['protein_per_100cal'] = (final_df['protein'] / cal_per_100).round(2)
        if 'fiber' in final_df.columns:
            final_df['fiber_per_100cal'] = (final_df['fiber'] / cal_per_100).round(2)
    
    print("Computed fields added")
    print()
    
    # === STEP 10: Data quality filtering ===
    print("STEP 10: Applying data quality filters...")
    print("-" * 80)
    
    rows_before = len(final_df)
    
    # Keep only foods with at least some nutrient data
    core_nutrient_cols = [col for col in ['calories_final', 'protein', 'fat', 'carbohydrate'] 
                         if col in final_df.columns]
    
    if core_nutrient_cols:
        final_df['nutrient_count'] = final_df[core_nutrient_cols].notna().sum(axis=1)
        final_df = final_df[final_df['nutrient_count'] >= 2].copy()
        final_df = final_df.drop('nutrient_count', axis=1)
    
    print(f"Removed {rows_before - len(final_df):,} foods with insufficient nutrient data")
    print(f"Final dataset: {len(final_df):,} foods")
    print()
    
    # === STEP 11: Write outputs ===
    print("STEP 11: Writing output files...")
    print("-" * 80)
    
    # Main output with all data - PARQUET
    output_main_parquet = os.path.join(OUTPUT_DIR, "restaurant_foods_nutrition.parquet")
    final_df.to_parquet(output_main_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"✓ Main output (Parquet): restaurant_foods_nutrition.parquet")
    print(f"  Size: {os.path.getsize(output_main_parquet) / 1024 / 1024:.2f} MB")
    print(f"  Rows: {len(final_df):,}")
    print(f"  Columns: {len(final_df.columns)}")
    
    # Main output with all data - CSV
    output_main_csv = os.path.join(OUTPUT_DIR, "restaurant_foods_nutrition.csv")
    final_df.to_csv(output_main_csv, index=False, encoding='utf-8')
    print(f"✓ Main output (CSV): restaurant_foods_nutrition.csv")
    print(f"  Size: {os.path.getsize(output_main_csv) / 1024 / 1024:.2f} MB")
    
    # Macros-only simplified output
    macro_cols = [
        'fdc_id', 'description', 'data_type', 'cooking_method', 'category_name',
        'calories_final', 'protein', 'fat', 'carbohydrate', 
        'fiber', 'sugar', 'sodium', 'cholesterol',
        'protein_pct', 'fat_pct', 'carb_pct',
        'serving_size_g_mean', 'portion_descriptions'
    ]
    macro_cols = [col for col in macro_cols if col in final_df.columns]
    
    # Macros - PARQUET
    output_macros_parquet = os.path.join(OUTPUT_DIR, "restaurant_foods_macros.parquet")
    final_df[macro_cols].to_parquet(output_macros_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"✓ Macros output (Parquet): restaurant_foods_macros.parquet")
    print(f"  Size: {os.path.getsize(output_macros_parquet) / 1024 / 1024:.2f} MB")
    
    # Macros - CSV
    output_macros_csv = os.path.join(OUTPUT_DIR, "restaurant_foods_macros.csv")
    final_df[macro_cols].to_csv(output_macros_csv, index=False, encoding='utf-8')
    print(f"✓ Macros output (CSV): restaurant_foods_macros.csv")
    print(f"  Size: {os.path.getsize(output_macros_csv) / 1024 / 1024:.2f} MB")
    
    # Metadata only
    metadata_cols_out = [
        'fdc_id', 'description', 'data_type', 'cooking_method', 'category_name',
        'publication_date', 'portion_descriptions'
    ]
    if 'brand_owner' in final_df.columns:
        metadata_cols_out.extend(['brand_owner', 'brand_name', 'branded_food_category'])
    if 'wweia_food_category_description' in final_df.columns:
        metadata_cols_out.append('wweia_food_category_description')
    
    metadata_cols_out = [col for col in metadata_cols_out if col in final_df.columns]
    
    # Metadata - PARQUET
    output_meta_parquet = os.path.join(OUTPUT_DIR, "restaurant_foods_metadata.parquet")
    final_df[metadata_cols_out].to_parquet(output_meta_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"✓ Metadata output (Parquet): restaurant_foods_metadata.parquet")
    print(f"  Size: {os.path.getsize(output_meta_parquet) / 1024 / 1024:.2f} MB")
    
    # Metadata - CSV
    output_meta_csv = os.path.join(OUTPUT_DIR, "restaurant_foods_metadata.csv")
    final_df[metadata_cols_out].to_csv(output_meta_csv, index=False, encoding='utf-8')
    print(f"✓ Metadata output (CSV): restaurant_foods_metadata.csv")
    print(f"  Size: {os.path.getsize(output_meta_csv) / 1024 / 1024:.2f} MB")
    
    # Summary by cooking method - CSV only
    if 'cooking_method' in final_df.columns and 'calories_final' in final_df.columns:
        cooking_summary = final_df.groupby('cooking_method').agg({
            'fdc_id': 'count',
            'calories_final': ['mean', 'median'],
            'protein': 'mean',
            'fat': 'mean',
            'carbohydrate': 'mean',
            'sodium': 'mean'
        }).round(2)
        
        cooking_summary.columns = ['count', 'avg_calories', 'median_calories', 
                                   'avg_protein_g', 'avg_fat_g', 'avg_carb_g', 'avg_sodium_mg']
        cooking_summary = cooking_summary.reset_index()
        cooking_summary = cooking_summary.sort_values('count', ascending=False)
        
        output_cooking_summary = os.path.join(OUTPUT_DIR, "cooking_method_summary.csv")
        cooking_summary.to_csv(output_cooking_summary, index=False)
        print(f"✓ Cooking method summary: cooking_method_summary.csv")
    
    print()
    print(f"Total output files created: 7 Parquet + 7 CSV = 14 files")
    
    # === Summary Statistics ===
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total restaurant foods: {len(final_df):,}")
    
    print(f"\nBy data type:")
    print(final_df['data_type'].value_counts().to_string())
    
    if 'category_name' in final_df.columns:
        print(f"\nTop 15 food categories:")
        print(final_df['category_name'].value_counts().head(15).to_string())
    
    if 'cooking_method' in final_df.columns:
        print(f"\nCooking method distribution:")
        print(final_df['cooking_method'].value_counts().to_string())
    
    print(f"\nNutrient coverage:")
    nutrient_check_cols = ['calories_final', 'protein', 'fat', 'carbohydrate', 
                          'fiber', 'sugar', 'sodium', 'cholesterol']
    for col in nutrient_check_cols:
        if col in final_df.columns:
            coverage = (final_df[col].notna().sum() / len(final_df) * 100)
            print(f"  {col:20s}: {coverage:5.1f}%")
    
    if 'calories_final' in final_df.columns:
        print(f"\nCalorie statistics:")
        print(f"  Mean: {final_df['calories_final'].mean():.1f} kcal")
        print(f"  Median: {final_df['calories_final'].median():.1f} kcal")
        print(f"  Min: {final_df['calories_final'].min():.1f} kcal")
        print(f"  Max: {final_df['calories_final'].max():.1f} kcal")
    
    # Add unit information to a separate file
    units_df = pd.DataFrame([
        {'nutrient': k, 'unit': v} 
        for k, v in nutrient_units.items()
    ])
    units_output = os.path.join(OUTPUT_DIR, "nutrient_units.csv")
    units_df.to_csv(units_output, index=False)
    print(f"\n✓ Nutrient units reference: nutrient_units.csv")
    
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

# Set seed for reproducibility so the numbers stay consistent every time you run it
np.random.seed(42)

# Define a massive dataset split (Total 10,000 rows)
num_optimal = 1000
num_trouble = 9000 

# 1. Trouble Data (Using raw sensor units: mg/kg, uS/cm, mV)
trouble_df = pd.DataFrame({
    'N_mg_per_kg': np.random.normal(50, 40, num_trouble).clip(0, 150),       
    'P_mg_per_kg': np.random.normal(25, 20, num_trouble).clip(0, 80),            
    'K_mg_per_kg': np.random.normal(20, 20, num_trouble).clip(0, 80),         
    'ph': np.random.normal(6.0, 1.5, num_trouble).clip(3.5, 9.0),   
    'EC_uS_cm': np.random.normal(1500, 1000, num_trouble).clip(50, 4000), 
    'ORP_mV': np.random.normal(0, 200, num_trouble).clip(-350, 350)
})

# 2. Optimal Data (Strictly locked into the v4 Ideal ranges)
optimal_df = pd.DataFrame({
    'N_mg_per_kg': np.random.uniform(70, 120, num_optimal),      
    'P_mg_per_kg': np.random.uniform(40, 60, num_optimal),       
    'K_mg_per_kg': np.random.uniform(35, 60, num_optimal),       
    'ph': np.random.uniform(5.5, 7.5, num_optimal),
    'EC_uS_cm': np.random.uniform(300, 2000, num_optimal),  
    'ORP_mV': np.random.uniform(-150, 150, num_optimal)
})

# Combine and shuffle the dataset so the AI doesn't memorize the order
df = pd.concat([trouble_df, optimal_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the v4 Logic Core
def calculate_all_outputs(row):
    
    # --- 1. HEALTH STATUS CLASSIFICATION ---
    # The order matters here; extreme physical dangers override simple nutrient deficiencies
    if row['EC_uS_cm'] > 2000:
        status = "High Salinity"
    elif row['ph'] < 5.5:
        status = "Acidic Danger"
    elif row['ph'] > 7.5:
        status = "Alkaline Lockout"
    elif row['ORP_mV'] < -150:
        status = "Iron Toxicity Risk"
    elif row['ORP_mV'] > 150:
        status = "Aerobic Risk (Dry)"
    elif row['EC_uS_cm'] < 300:
        status = "Severe Depletion"
    elif row['N_mg_per_kg'] < 70 or row['P_mg_per_kg'] < 40 or row['K_mg_per_kg'] < 35:
        status = "Nutrient Deficient"
    else:
        status = "Optimal"
        
    # --- 2. FERTILIZER AMOUNTS (Targeting v4 Baselines: 70 N, 40 P, 35 K) ---
    p_def = max(0, 40 - row['P_mg_per_kg'])
    dap_kg = round(p_def / 0.46, 1) if p_def > 0 else 0.0
    
    n_provided_by_dap = dap_kg * 0.18
    n_def = max(0, 70 - row['N_mg_per_kg'] - n_provided_by_dap)
    urea_kg = round(n_def / 0.46, 1) if n_def > 0 else 0.0
    
    k_def = max(0, 35 - row['K_mg_per_kg'])
    mop_kg = round(k_def / 0.60, 1) if k_def > 0 else 0.0

    # --- 3. LOW EC FERTILIZER BOOST ---
    low_ec_boost_kg = 0.0
    if row['EC_uS_cm'] < 300:
        low_ec_boost_kg = round((300 - row['EC_uS_cm']) * 0.1, 1) 

    # --- 4. WATER MANAGEMENT (Liters) ---
    if row['EC_uS_cm'] > 2000:
        phase1_water_L = round((row['EC_uS_cm'] - 2000) * 400, 0) 
    else:
        phase1_water_L = 0.0
        
    if row['ORP_mV'] > 150:
        phase2_water_L = 10000.0
    else:
        phase2_water_L = 0.0
        
    # --- 5. LIME / GYPSUM AMENDMENTS ---
    lime_kg = 0.0
    gypsum_kg = 0.0
    
    if row['ph'] < 5.5:
        lime_kg = round(max(50.0, (6.5 - row['ph']) * 150), 1)
        
    if row['ph'] > 7.5:
        gypsum_kg = round(max(50.0, (row['ph'] - 7.0) * 100.0), 1)

    return pd.Series([
        status, 
        urea_kg, dap_kg, mop_kg, low_ec_boost_kg,
        phase1_water_L, phase2_water_L, 
        lime_kg, gypsum_kg
    ])

# Explicitly name the output columns with their units
output_cols = [
    'Health_Status', 
    'Urea_kg_per_acre', 'DAP_kg_per_acre', 'MOP_kg_per_acre', 'Low_EC_Fertilizer_Boost_kg',
    'Phase1_EC_Flush_Water_Liters', 'Phase2_ORP_Flood_Water_Liters', 
    'Lime_kg_per_acre', 'Gypsum_kg_per_acre'
]

# Process the data
df[output_cols] = df.apply(calculate_all_outputs, axis=1)

# Reorder columns to ensure inputs and outputs are clearly separated
final_column_order = [
    'N_mg_per_kg', 'P_mg_per_kg', 'K_mg_per_kg', 'ph', 'EC_uS_cm', 'ORP_mV', 
    'Health_Status', 
    'Urea_kg_per_acre', 'DAP_kg_per_acre', 'MOP_kg_per_acre', 'Low_EC_Fertilizer_Boost_kg',
    'Phase1_EC_Flush_Water_Liters', 'Phase2_ORP_Flood_Water_Liters', 
    'Lime_kg_per_acre', 'Gypsum_kg_per_acre'
]

df = df[final_column_order]

# Save the massive dataset
file_name = "KrishiLink_10k_RawUnits_Training_Data.csv"
df.to_csv(file_name, index=False)

print(f"Success! {file_name} generated with 10,000 rows.")
print("\n--- FINAL DATASET DISTRIBUTION ---")
print(df['Health_Status'].value_counts())
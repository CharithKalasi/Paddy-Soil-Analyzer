"""Detect deterministic / rule-derived targets in the dataset.
Generates `leakage_report.json` in the working directory.
"""
import json
import pandas as pd
import numpy as np


def run_check(csv_path="KrishiLink_10k_RawUnits_Training_Data.csv", out_path="leakage_report.json"):
    df = pd.read_csv(csv_path)
    report = {}
    report['shape'] = df.shape
    report['columns'] = list(df.columns)
    num = df.select_dtypes(include=[np.number])
    report['numeric_columns'] = list(num.columns)

    # quick high-correlation pairs
    high_pairs = []
    cols = num.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            r = float(num[a].corr(num[b]))
            if abs(r) > 0.95:
                high_pairs.append({'col_a': a, 'col_b': b, 'r': round(r,4)})
    report['high_numeric_pairs_abs_gt_0.95'] = high_pairs

    # attempt to recompute outputs using known generator logic
    in_cols = ['N_mg_per_kg','P_mg_per_kg','K_mg_per_kg','ph','EC_uS_cm','ORP_mV']
    missing = [c for c in in_cols if c not in df.columns]
    report['missing_input_columns'] = missing

    if not missing:
        def calc(row):
            N = row['N_mg_per_kg']; P = row['P_mg_per_kg']; K = row['K_mg_per_kg']
            ph = row['ph']; EC = row['EC_uS_cm']; ORP = row['ORP_mV']
            if EC > 2000:
                status = "High Salinity"
            elif ph < 5.5:
                status = "Acidic Danger"
            elif ph > 7.5:
                status = "Alkaline Lockout"
            elif ORP < -150:
                status = "Iron Toxicity Risk"
            elif ORP > 150:
                status = "Aerobic Risk (Dry)"
            elif EC < 300:
                status = "Severe Depletion"
            elif N < 70 or P < 40 or K < 35:
                status = "Nutrient Deficient"
            else:
                status = "Optimal"
            p_def = max(0, 40 - P)
            dap_kg = round(p_def / 0.46, 1) if p_def > 0 else 0.0
            n_provided_by_dap = dap_kg * 0.18
            n_def = max(0, 70 - N - n_provided_by_dap)
            urea_kg = round(n_def / 0.46, 1) if n_def > 0 else 0.0
            k_def = max(0, 35 - K)
            mop_kg = round(k_def / 0.60, 1) if k_def > 0 else 0.0
            low_ec_boost_kg = 0.0
            if EC < 300:
                low_ec_boost_kg = round((300 - EC) * 0.1, 1)
            if EC > 2000:
                phase1_water_L = round((EC - 2000) * 400, 0)
            else:
                phase1_water_L = 0.0
            if ORP > 150:
                phase2_water_L = 10000.0
            else:
                phase2_water_L = 0.0
            lime_kg = 0.0; gypsum_kg = 0.0
            if ph < 5.5:
                lime_kg = round(max(50.0, (6.5 - ph) * 150), 1)
            if ph > 7.5:
                gypsum_kg = round(max(50.0, (ph - 7.0) * 100.0), 1)
            return pd.Series([
                status, urea_kg, dap_kg, mop_kg, low_ec_boost_kg,
                phase1_water_L, phase2_water_L, lime_kg, gypsum_kg
            ])

        out_cols = [
            'Health_Status', 'Urea_kg_per_acre','DAP_kg_per_acre','MOP_kg_per_acre',
            'Low_EC_Fertilizer_Boost_kg','Phase1_EC_Flush_Water_Liters','Phase2_ORP_Flood_Water_Liters',
            'Lime_kg_per_acre','Gypsum_kg_per_acre'
        ]
        recomputed = df[in_cols].apply(calc, axis=1)
        recomputed.columns = ['Health_Status_calc','Urea_calc','DAP_calc','MOP_calc','LowEC_calc','Phase1_calc','Phase2_calc','Lime_calc','Gypsum_calc']

        mapping = {
            'Health_Status': 'Health_Status_calc',
            'Urea_kg_per_acre': 'Urea_calc',
            'DAP_kg_per_acre': 'DAP_calc',
            'MOP_kg_per_acre': 'MOP_calc',
            'Low_EC_Fertilizer_Boost_kg': 'LowEC_calc',
            'Phase1_EC_Flush_Water_Liters': 'Phase1_calc',
            'Phase2_ORP_Flood_Water_Liters': 'Phase2_calc',
            'Lime_kg_per_acre': 'Lime_calc',
            'Gypsum_kg_per_acre': 'Gypsum_calc'
        }
        compare = {}
        for saved, calccol in mapping.items():
            if saved not in df.columns:
                compare[saved] = {'status': 'missing_in_csv'}
                continue
            s = df[saved]
            c = recomputed[calccol]
            # object/string compare
            if s.dtype == object or s.dtype == 'string' or s.dtype == 'str':
                compare[saved] = {'equal_fraction': float((s == c).mean())}
            else:
                s_num = pd.to_numeric(s, errors='coerce')
                c_num = pd.to_numeric(c, errors='coerce')
                mask = s_num.notna() & c_num.notna()
                if mask.sum() == 0:
                    compare[saved] = {'equal_fraction': 0.0, 'compared_count': 0}
                else:
                    close = np.isclose(s_num[mask].values, c_num[mask].values, atol=1e-6)
                    compare[saved] = {'equal_fraction': float(close.mean()), 'compared_count': int(mask.sum())}
        report['recompute_comparison'] = compare

    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote leakage report to: {out_path}")


if __name__ == '__main__':
    run_check()

from predict import phase1_predict, phase2_predict


TEST_READINGS = [
    {"N": 10, "P": 5, "K": 8, "ph": 4.0, "EC_uS_cm": 200, "ORP_mV": -200},
    {"N": 60, "P": 30, "K": 25, "ph": 8.5, "EC_uS_cm": 2800, "ORP_mV": 300},
    {"N": 50, "P": 25, "K": 20, "ph": 6.5, "EC_uS_cm": 800, "ORP_mV": -50},
    {"N": 0, "P": 0, "K": 0, "ph": 3.5, "EC_uS_cm": 50, "ORP_mV": -350},
    {"N": 80, "P": 50, "K": 45, "ph": 9.0, "EC_uS_cm": 3500, "ORP_mV": 350},
]


def print_phase1_result(result):
    print("Phase 1 Recommendations")
    print("-" * 24)
    print("NPK:")
    print(f"  Urea_kg_per_acre: {result['NPK']['Urea_kg_per_acre']:.2f}")
    print(f"  DAP_kg_per_acre: {result['NPK']['DAP_kg_per_acre']:.2f}")
    print(f"  MOP_kg_per_acre: {result['NPK']['MOP_kg_per_acre']:.2f}")
    print("PH:")
    print(f"  Lime_kg_per_acre: {result['PH']['Lime_kg_per_acre']:.2f}")
    print(f"  Gypsum_kg_per_acre: {result['PH']['Gypsum_kg_per_acre']:.2f}")
    print("EC:")
    print(f"  Low_EC_Fertilizer_Boost_kg: {result['EC']['Low_EC_Fertilizer_Boost_kg']:.2f}")
    print(f"  Phase1_EC_Flush_Water_Liters: {result['EC']['Phase1_EC_Flush_Water_Liters']:.2f}")


def print_phase2_result(result):
    print("Phase 2 Recommendation")
    print("-" * 24)
    print(f"Phase2_ORP_Flood_Water_Liters: {result['Phase2_ORP_Flood_Water_Liters']:.2f}")


def main():
    for index, reading in enumerate(TEST_READINGS, start=1):
        print(f"Test Case {index}")
        print("=" * 12)
        print(
            f"Sensor Input: N={reading['N']}, P={reading['P']}, K={reading['K']}, "
            f"ph={reading['ph']}, EC_uS_cm={reading['EC_uS_cm']}, ORP_mV={reading['ORP_mV']}"
        )
        print()

        phase1_result = phase1_predict(
            N=reading["N"],
            P=reading["P"],
            K=reading["K"],
            ph=reading["ph"],
            EC=reading["EC_uS_cm"],
        )
        print_phase1_result(phase1_result)
        print()

        phase2_result = phase2_predict(ORP=reading["ORP_mV"])
        print_phase2_result(phase2_result)
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
# ============================================================
# GRIDGUARD AI
# MODEL 1 - TEMPERATURE ANOMALY DETECTION
# FINAL DEPLOYMENT CONFIGURATION
# ============================================================

# Model input
WINDOW_SIZE = 20

# Sampling interval
SAMPLING_TIME = 1.0  # seconds

# Anomaly decision threshold
ANOMALY_THRESHOLD = 0.45

# Temperature model
BASE_TEMPERATURE = 29.0

# Thermistor parameters
SERIES_RESISTOR = 10500.0
NOMINAL_RESISTANCE = 10000.0
NOMINAL_TEMPERATURE = 25.0
BETA = 3950.0

# ADC parameters
ADC_MAX = 4095.0
VREF = 3.30
ADC_CORRECTION = 1.34

# Optional temperature correction
TEMP_OFFSET = 0.0


if __name__ == "__main__":
    print("==========================================")
    print(" GRIDGUARD AI - MODEL 1 CONFIGURATION")
    print("==========================================")
    print(f"Window size       : {WINDOW_SIZE}")
    print(f"Sampling time     : {SAMPLING_TIME} seconds")
    print(f"Anomaly threshold : {ANOMALY_THRESHOLD}")
    print("==========================================")
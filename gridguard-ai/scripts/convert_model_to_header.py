from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.tflite"

OUTPUT_PATH = PROJECT_ROOT / "models" / "anomaly_model.h"


# ============================================================
# CHECK MODEL EXISTS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "\nPut your .tflite model in the models folder "
        "or update MODEL_PATH."
    )


# ============================================================
# READ MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model_data = f.read()


# ============================================================
# CONVERT TO C HEADER
# ============================================================

with open(OUTPUT_PATH, "w") as f:

    f.write("#ifndef ANOMALY_MODEL_H\n")
    f.write("#define ANOMALY_MODEL_H\n\n")

    f.write(
        "const unsigned char anomaly_model_tflite[] = {\n"
    )

    for i, byte in enumerate(model_data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i < len(model_data) - 1:
            f.write(",")
            if (i + 1) % 12 != 0:
                f.write(" ")

        if (i + 1) % 12 == 0:
            f.write("\n")

    if len(model_data) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(
        "const unsigned int anomaly_model_tflite_len = "
    )

    f.write(f"{len(model_data)};\n\n")

    f.write("#endif\n")


# ============================================================
# RESULT
# ============================================================

print("=" * 60)
print("MODEL CONVERSION COMPLETE")
print("=" * 60)

print(f"\nInput model:")
print(MODEL_PATH)

print(f"\nOutput header:")
print(OUTPUT_PATH)

print(f"\nModel size:")
print(f"{len(model_data):,} bytes")
print(f"{len(model_data) / 1024:.2f} KB")

print("\nThe .h file is ready for ESP32 deployment.")
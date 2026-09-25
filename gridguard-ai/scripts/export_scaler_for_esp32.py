import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


mean = np.load(config.SCALER_MEAN_FILE)
scale = np.load(config.SCALER_SCALE_FILE)


print("\n" + "=" * 60)
print("GRIDGUARD AI - ESP32 SCALER EXPORT")
print("=" * 60)


print("\n// Paste these into your ESP32 code\n")


print("const float SCALER_MEAN[20] = {")

for i, value in enumerate(mean):

    if i % 5 == 0:
        print("    ", end="")

    print(f"{value:.8f}f", end="")

    if i < len(mean) - 1:
        print(", ", end="")

    if i % 5 == 4:
        print()

if len(mean) % 5 != 0:
    print()

print("};")


print("\nconst float SCALER_SCALE[20] = {")

for i, value in enumerate(scale):

    if i % 5 == 0:
        print("    ", end="")

    print(f"{value:.8f}f", end="")

    if i < len(scale) - 1:
        print(", ", end="")

    if i % 5 == 4:
        print()

if len(scale) % 5 != 0:
    print()

print("};")


print("\n" + "=" * 60)
print("RAW VALUES")
print("=" * 60)

print("\nMean:")
print(mean)

print("\nScale:")
print(scale)
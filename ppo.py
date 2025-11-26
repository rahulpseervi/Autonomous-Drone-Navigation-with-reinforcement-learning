import os

MODEL_PATH = "ppo_drone_2d.zip"

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"Deleted existing model file: {MODEL_PATH}")
else:
    print(f"Model file not found: {MODEL_PATH}. No deletion needed.")

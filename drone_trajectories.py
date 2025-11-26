import os # Added this import to ensure 'os' is defined in this cell

if __name__ == "__main__":
    MODEL_PATH = "ppo_drone_2d.zip"
    # If model exists, skip training
    if not os.path.exists(MODEL_PATH):
        # Adjust timesteps depending on your machine
        print("\n--- Installing dependencies including shimmy ---")
        # The original code provided the dependencies in the docstring, updating the install to ensure shimmy is present.
        !pip install gym==0.26.2 stable-baselines3[extra] numpy matplotlib shimmy>=0.2.1 # shimmy>=0.2.1 due to SB3 v2.3.0 requirement.
        print("\n--- Dependencies installed ---\n")
        trained_model = train_and_save(model_path=MODEL_PATH, timesteps=1_000_000)
    else:
        print("Found existing model:", MODEL_PATH)

    # Evaluate + plot
    evaluate_and_plot(model_path=MODEL_PATH, episodes=15)

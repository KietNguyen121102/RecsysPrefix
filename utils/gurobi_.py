import gurobipy as gp
import sys

try:
    # Attempt to create an environment, which requires a valid license
    env = gp.Env() 
    # Optional: print the license file path if successful
    print(f"Using license file: {env.getParam('LicenseFile')}") 
    print("Gurobi license is valid and found.")

    # You can then proceed to create a model and work with Gurobi
    m = gp.Model(env=env)

except gp.GurobiError as e:
    print(f"Gurobi Error: {e.message}")
    print("Could not obtain a Gurobi license. Please ensure your license is set up correctly.")
    print("Refer to the 
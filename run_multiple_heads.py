import subprocess
import sys
import os

# Define a list of head values to test
head_values = [1, 2, 4]

# Get the directory of the current script to build the path to test_sdcn_dlaa_NEW.py
current_dir = os.path.dirname(os.path.abspath(__file__))
test_script_path = os.path.join(current_dir, "test_sdcn_dlaa_NEW.py")

# Check if test_sdcn_dlaa_NEW.py exists
if not os.path.exists(test_script_path):
    print(f"Error: Script 'test_sdcn_dlaa_NEW.py' not found at expected location: {test_script_path}")
    sys.exit(1)

# Get the path of the current Python interpreter
python_executable = sys.executable

print("Starting tests with different head values...")
print("=" * 30)

for heads in head_values:
    print(f"\n>>> Running test with heads = {heads}...")
    print("-" * 20)

    # Build command line arguments
    # Note: We assume test_sdcn_dlaa_NEW.py uses argparse to handle other necessary parameters
    # If test_sdcn_dlaa_NEW.py requires additional parameters, add them here
    command = [
        python_executable,
        test_script_path,
        f"--heads={heads}"
        # If other parameters are needed, add them like this:
        # "--lr=0.001",
        # "--n_clusters=3",
        # ...
    ]

    try:
        # Run subprocess and print output directly to terminal
        # check=True will raise CalledProcessError if subprocess returns non-zero exit code
        result = subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"\n--- Test with heads = {heads} completed ---")

    except subprocess.CalledProcessError as e:
        print(f"\n--- Test with heads = {heads} failed ---")
        print(f"Error message: {e}")
        # Can choose to stop here or continue testing next heads value
        # continue
        break # Stop if one fails

    except FileNotFoundError:
        print(f"Error: Cannot find Python interpreter '{python_executable}' or script '{test_script_path}'")
        break

    print("=" * 30)

print("\nAll tests completed.")
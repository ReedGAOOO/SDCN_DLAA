import subprocess
import sys
import os

# --- Configuration ---
# Target script name to execute (modify to point to the AMP version test script)
target_script = "test_sdcn_dlaa_NEW_amp.py"

# List of heads parameters to test
heads_to_test = [1, 2, 4]

# Other arguments you may want to pass fixed to target_script (if needed)
# Example: other_args = ["--lr", "0.0005", "--n_clusters", "5"]
# If empty list, target_script will use its internally defined default values (except --heads)
other_args = []
# --- End Configuration ---

# Check if target script exists
if not os.path.exists(target_script):
    print(f"Error: Target script '{target_script}' not found in current directory.")
    print("Please ensure this script is in the same directory as the target test script and the target test script has been created.")
    sys.exit(1) # Exit script

print(f"Starting batch test script: {target_script}")
print(f"Will test with following heads values: {heads_to_test}")
print("-" * 60)

# Store each run's status (success/failure)
run_statuses = {}

for heads_value in heads_to_test:
    print(f"\n=======>>>>> Starting test: heads = {heads_value} <<<<<=======")

    # Build command line arguments list
    # Use sys.executable to ensure using the same Python interpreter running this script
    command = [sys.executable, target_script, "--heads", str(heads_value)]

    # Add other fixed arguments (if configured)
    if other_args:
        command.extend(other_args)

    print(f"Executing command: {' '.join(command)}")
    print("-" * 60)

    try:
        # Execute subprocess
        # check=True: If subprocess returns non-zero exit code (indicating error), raises CalledProcessError
        # text=True: (Python 3.7+) Makes stdout and stderr be treated as text
        # stdout/stderr will inherit from parent process by default, so output goes directly to current terminal
        result = subprocess.run(command, check=True, text=True)
        print("-" * 60)
        print(f"=======>>>>> Test completed: heads = {heads_value} (Success) <<<<<=======\n")
        run_statuses[heads_value] = "Success"

    except subprocess.CalledProcessError as e:
        # If subprocess execution fails
        print("-" * 60)
        print(f"=======>>>>> Test failed: heads = {heads_value} (Command returned error code: {e.returncode}) <<<<<=======")
        # Specific error output should have been printed to terminal by subprocess
        run_statuses[heads_value] = f"Failed (Error code: {e.returncode})"
    except KeyboardInterrupt:
        # If user manually interrupts (Ctrl+C)
        print("\nUser interrupted testing.")
        run_statuses[heads_value] = "User interrupted"
        break # Stop subsequent tests
    except Exception as e:
        # Catch other potential errors
        print(f"Unexpected error occurred during test (heads={heads_value}): {e}")
        run_statuses[heads_value] = f"Failed (Unexpected error: {e})"


print("\n==================== Batch Test Summary ====================")
for heads_value, status in run_statuses.items():
    print(f"Heads = {heads_value}: {status}")
print("======================================================")
print("All test runs completed.")
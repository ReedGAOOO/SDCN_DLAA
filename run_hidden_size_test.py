import subprocess
import sys
import os

# --- Configuration ---
# Target script name to execute
# Updated to point to hiddensize version
target_script = "test_sdcn_dlaa_NEW_hiddensize.py"

# Fixed heads parameter value
fixed_heads = 1

# Hidden size profiles to test: [hs1, hs2, hs3]
# hs1 -> n_enc_1, n_dec_3
# hs2 -> n_enc_2, n_dec_2
# hs3 -> n_enc_3, n_dec_1
hidden_size_profiles = [
    [256, 256, 256],
    [500, 500, 512],
    [256, 256, 1024],
    [500, 500, 1024],
]

# Other arguments you may want to fix and pass to target_script (if needed)
# Example: other_args = ["--lr", "0.0005", "--n_clusters", "5"]
# If empty list, target_script will use its internally defined default values (except for --heads and --hs*)
other_args = []
# --- End Configuration ---

# Check if target script exists
# Updated check logic to accommodate new filename
if not os.path.exists(target_script):
    print(f"Error: Target script '{target_script}' not found in current directory.")
    print("Please ensure this script is in the same directory as the target script.")
    print("Also, make sure the target script has been modified to accept --hs1, --hs2, --hs3 parameters.")
    sys.exit(1) # 退出脚本

print(f"Starting batch testing of script: {target_script}")
print(f"Will fix heads = {fixed_heads}")
print(f"Will test with the following hidden size profiles [hs1, hs2, hs3]:")
for profile in hidden_size_profiles:
    print(f"  - {profile}")
print("-" * 60)

# Store status (success/failure) of each run
run_statuses = {}

for profile in hidden_size_profiles:
    hs1, hs2, hs3 = profile
    profile_str = f"hs=[{hs1},{hs2},{hs3}]"

    print(f"\n=======>>>>> Starting test: {profile_str}, heads = {fixed_heads} <<<<<=======")

    # Build command line arguments list
    # Use sys.executable to ensure using the same Python interpreter running this script
    command = [
        sys.executable,
        target_script, # 已更新为 hiddensize 版本
        "--heads", str(fixed_heads),
        "--hs1", str(hs1),
        "--hs2", str(hs2),
        "--hs3", str(hs3),
    ]

    # 添加其他固定参数 (如果配置了)
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
        print(f"=======>>>>> Test completed: {profile_str}, heads = {fixed_heads} (Success) <<<<<=======\n")
        run_statuses[profile_str] = "Success"

    except subprocess.CalledProcessError as e:
        # If subprocess execution fails
        print("-" * 60)
        print(f"=======>>>>> Test failed: {profile_str}, heads = {fixed_heads} (Command returned error code: {e.returncode}) <<<<<=======")
        # 具体的错误输出应该已经由子进程打印到终端了
        run_statuses[profile_str] = f"Failed (Error code: {e.returncode})"
    except KeyboardInterrupt:
        # If user manually interrupts (Ctrl+C)
        print("\nUser interrupted testing.")
        run_statuses[profile_str] = "User interrupted"
        break # 停止后续测试
    except Exception as e:
        # Catch other potential errors
        print(f"Unexpected error occurred during test ({profile_str}, heads={fixed_heads}): {e}")
        run_statuses[profile_str] = f"Failed (Unexpected error: {e})"


print("\n==================== Batch Test Summary ====================")
for profile_str, status in run_statuses.items():
    print(f"{profile_str}, heads = {fixed_heads}: {status}")
print("======================================================")
print("All test runs completed.")
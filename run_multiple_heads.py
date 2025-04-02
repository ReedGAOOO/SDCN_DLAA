import subprocess
import sys
import os

# 定义要测试的 heads 值列表
head_values = [1, 2, 4]

# 获取当前脚本所在的目录，用于构建 test_sdcn_dlaa_NEW.py 的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
test_script_path = os.path.join(current_dir, "test_sdcn_dlaa_NEW.py")

# 检查 test_sdcn_dlaa_NEW.py 是否存在
if not os.path.exists(test_script_path):
    print(f"错误: 脚本 'test_sdcn_dlaa_NEW.py' 在预期位置未找到: {test_script_path}")
    sys.exit(1)

# 获取当前 Python 解释器的路径
python_executable = sys.executable

print("开始运行不同 heads 值的测试...")
print("=" * 30)

for heads in head_values:
    print(f"\n>>> 正在运行 heads = {heads} 的测试...")
    print("-" * 20)

    # 构建命令行参数
    # 注意：我们假设 test_sdcn_dlaa_NEW.py 使用 argparse 处理其他必要的参数
    # 如果 test_sdcn_dlaa_NEW.py 需要其他参数，请在这里添加
    command = [
        python_executable,
        test_script_path,
        f"--heads={heads}"
        # 如果需要其他参数，像这样添加:
        # "--lr=0.001",
        # "--n_clusters=3",
        # ...
    ]

    try:
        # 运行子进程，并将输出直接打印到终端
        # check=True 会在子进程返回非零退出码时抛出 CalledProcessError
        result = subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"\n--- heads = {heads} 测试完成 ---")

    except subprocess.CalledProcessError as e:
        print(f"\n--- heads = {heads} 测试失败 ---")
        print(f"错误信息: {e}")
        # 可以选择在这里停止，或者继续测试下一个 heads 值
        # continue
        break # 如果一个失败就停止

    except FileNotFoundError:
        print(f"错误: 无法找到 Python 解释器 '{python_executable}' 或脚本 '{test_script_path}'")
        break

    print("=" * 30)

print("\n所有测试运行完毕。")
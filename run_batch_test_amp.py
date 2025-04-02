import subprocess
import sys
import os

# --- 配置 ---
# 要执行的目标脚本名称 (修改为指向 AMP 版本的测试脚本)
target_script = "test_sdcn_dlaa_NEW_amp.py"

# 要测试的 heads 参数列表
heads_to_test = [1, 2, 4]

# 其他你可能想要固定传递给 target_script 的参数 (如果需要的话)
# 例如: other_args = ["--lr", "0.0005", "--n_clusters", "5"]
# 如果为空列表，则 target_script 将使用其内部定义的默认值 (除了 --heads)
other_args = []
# --- 结束配置 ---

# 检查目标脚本是否存在
if not os.path.exists(target_script):
    print(f"错误：目标脚本 '{target_script}' 未在当前目录中找到。")
    print("请确保此脚本与目标测试脚本在同一目录下，并且目标测试脚本已创建。")
    sys.exit(1) # 退出脚本

print(f"开始批量测试脚本: {target_script}")
print(f"将依次使用以下 heads 值进行测试: {heads_to_test}")
print("-" * 60)

# 存储每个运行的状态 (成功/失败)
run_statuses = {}

for heads_value in heads_to_test:
    print(f"\n=======>>>>> 开始测试: heads = {heads_value} <<<<<=======")

    # 构建命令行参数列表
    # 使用 sys.executable 确保使用的是当前运行此脚本的 Python 解释器
    command = [sys.executable, target_script, "--heads", str(heads_value)]

    # 添加其他固定参数 (如果配置了)
    if other_args:
        command.extend(other_args)

    print(f"执行命令: {' '.join(command)}")
    print("-" * 60)

    try:
        # 执行子进程
        # check=True: 如果子进程返回非零退出码 (表示错误)，则会引发 CalledProcessError
        # text=True: (Python 3.7+) 使 stdout 和 stderr 作为文本处理
        # stdout/stderr 默认会继承父进程，所以会直接打印到当前终端
        result = subprocess.run(command, check=True, text=True)
        print("-" * 60)
        print(f"=======>>>>> 测试完成: heads = {heads_value} (成功) <<<<<=======\n")
        run_statuses[heads_value] = "成功"

    except subprocess.CalledProcessError as e:
        # 如果子进程执行出错
        print("-" * 60)
        print(f"=======>>>>> 测试失败: heads = {heads_value} (命令返回错误码: {e.returncode}) <<<<<=======")
        # 具体的错误输出应该已经由子进程打印到终端了
        run_statuses[heads_value] = f"失败 (错误码: {e.returncode})"
    except KeyboardInterrupt:
        # 如果用户手动中断 (Ctrl+C)
        print("\n用户中断了测试。")
        run_statuses[heads_value] = "用户中断"
        break # 停止后续测试
    except Exception as e:
        # 捕捉其他潜在错误
        print(f"执行测试时发生意外错误 (heads={heads_value}): {e}")
        run_statuses[heads_value] = f"失败 (意外错误: {e})"


print("\n==================== 批量测试总结 ====================")
for heads_value, status in run_statuses.items():
    print(f"Heads = {heads_value}: {status}")
print("======================================================")
print("所有测试运行完毕。")
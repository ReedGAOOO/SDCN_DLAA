import subprocess
import sys
import os

# --- 配置 ---
# 要执行的目标脚本名称
# 已更新为指向 hiddensize 版本
target_script = "test_sdcn_dlaa_NEW_hiddensize.py"

# 固定 heads 参数值
fixed_heads = 1

# 要测试的 hidden size profile 列表: [hs1, hs2, hs3]
# hs1 -> n_enc_1, n_dec_3
# hs2 -> n_enc_2, n_dec_2
# hs3 -> n_enc_3, n_dec_1
hidden_size_profiles = [
    [256, 256, 256],
    [500, 500, 512],
    [256, 256, 1024],
    [500, 500, 1024],
]

# 其他你可能想要固定传递给 target_script 的参数 (如果需要的话)
# 例如: other_args = ["--lr", "0.0005", "--n_clusters", "5"]
# 如果为空列表，则 target_script 将使用其内部定义的默认值 (除了 --heads 和 --hs*)
other_args = []
# --- 结束配置 ---

# 检查目标脚本是否存在
# 已更新检查逻辑以适应新文件名
if not os.path.exists(target_script):
    print(f"错误：目标脚本 '{target_script}' 未在当前目录中找到。")
    print("请确保此脚本与目标脚本在同一目录下。")
    print("并且，请确保目标脚本已被修改以接受 --hs1, --hs2, --hs3 参数。")
    sys.exit(1) # 退出脚本

print(f"开始批量测试脚本: {target_script}")
print(f"将固定 heads = {fixed_heads}")
print(f"将依次使用以下 hidden size profiles [hs1, hs2, hs3] 进行测试:")
for profile in hidden_size_profiles:
    print(f"  - {profile}")
print("-" * 60)

# 存储每个运行的状态 (成功/失败)
run_statuses = {}

for profile in hidden_size_profiles:
    hs1, hs2, hs3 = profile
    profile_str = f"hs=[{hs1},{hs2},{hs3}]"

    print(f"\n=======>>>>> 开始测试: {profile_str}, heads = {fixed_heads} <<<<<=======")

    # 构建命令行参数列表
    # 使用 sys.executable 确保使用的是当前运行此脚本的 Python 解释器
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

    print(f"执行命令: {' '.join(command)}")
    print("-" * 60)

    try:
        # 执行子进程
        # check=True: 如果子进程返回非零退出码 (表示错误)，则会引发 CalledProcessError
        # text=True: (Python 3.7+) 使 stdout 和 stderr 作为文本处理
        # stdout/stderr 默认会继承父进程，所以会直接打印到当前终端
        result = subprocess.run(command, check=True, text=True)
        print("-" * 60)
        print(f"=======>>>>> 测试完成: {profile_str}, heads = {fixed_heads} (成功) <<<<<=======\n")
        run_statuses[profile_str] = "成功"

    except subprocess.CalledProcessError as e:
        # 如果子进程执行出错
        print("-" * 60)
        print(f"=======>>>>> 测试失败: {profile_str}, heads = {fixed_heads} (命令返回错误码: {e.returncode}) <<<<<=======")
        # 具体的错误输出应该已经由子进程打印到终端了
        run_statuses[profile_str] = f"失败 (错误码: {e.returncode})"
    except KeyboardInterrupt:
        # 如果用户手动中断 (Ctrl+C)
        print("\n用户中断了测试。")
        run_statuses[profile_str] = "用户中断"
        break # 停止后续测试
    except Exception as e:
        # 捕捉其他潜在错误
        print(f"执行测试时发生意外错误 ({profile_str}, heads={fixed_heads}): {e}")
        run_statuses[profile_str] = f"失败 (意外错误: {e})"


print("\n==================== 批量测试总结 ====================")
for profile_str, status in run_statuses.items():
    print(f"{profile_str}, heads = {fixed_heads}: {status}")
print("======================================================")
print("所有测试运行完毕。")
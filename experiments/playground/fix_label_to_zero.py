import os

def fix_labels_to_zero(folder):
    """
    批量将 YOLO 标签文件中的类别 ID 改为 0
    folder: 存放 .txt 标签文件的目录
    """
    count_files = 0
    count_lines = 0

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # 把第一个数字改成 0
                    new_lines.append(" ".join(parts) + "\n")
                    count_lines += 1
                else:
                    # 跳过空行或格式异常的行
                    new_lines.append(line)

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            count_files += 1

    print(f"✅ 已修改 {count_files} 个文件，共 {count_lines} 行标签为 0。")

if __name__ == "__main__":
    folder = input("请输入标签文件所在的文件夹路径: ").strip()
    if os.path.isdir(folder):
        fix_labels_to_zero(folder)
    else:
        print("❌ 路径不存在，请检查输入。")

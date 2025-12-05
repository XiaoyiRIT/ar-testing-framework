import pandas as pd
import unicodedata

df = pd.read_csv("data_stat.csv")

print("=== 原始唯一值列表（repr 显示隐藏字符）===\n")
unique_vals = df["App"].astype(str).unique()
for v in unique_vals:
    print(repr(v))

print("\n=== 清洗前，每个 App 值的计数 ===\n")
print(df["App"].astype(str).value_counts())

print("\n=== 显示每个 App 字符的 Unicode 码位（帮助定位隐藏字符）===\n")

def show_unicode(s):
    return " ".join(f"{c}({ord(c):04X})" for c in s)

for v in unique_vals:
    print(f"{repr(v)} → {show_unicode(v)}")

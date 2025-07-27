import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 主题风格
sns.set(style='whitegrid', font_scale=1.1)

# 读取数据
df = pd.read_csv("build/result/pca_result.csv")

# 类型转换
df['label'] = df['label'].astype(int)

# 添加可读列
df['ground_truth'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})

# 定义调色板（现代配色）
label_palette = {'Normal': '#4CAF50', 'Anomaly': '#F44336'}

# 创建图形
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='ground_truth',
    style='ground_truth',
    palette=label_palette,
    s=10,
    edgecolor='black',
    linewidth=0.001,
    alpha=0.25
)

plt.title("Ground Truth (Normal vs Anomaly)", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图像
plt.savefig("pca.png", dpi=300)
print("✅ 图像保存为 pca.png")
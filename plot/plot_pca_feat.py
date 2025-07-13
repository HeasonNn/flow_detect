import pandas as pd
import matplotlib.pyplot as plt

# 读取 PCA + label 数据
df = pd.read_csv("../build/pca_result.csv", header=None, names=["PCA1", "PCA2", "IsAnomaly"])

# 拆分正常 vs 异常
normal = df[df["IsAnomaly"] == 0]
anomaly = df[df["IsAnomaly"] == 1]

# 创建图像
plt.figure(figsize=(10, 8))

# 正常样本：浅蓝色小点
plt.scatter(normal["PCA1"], normal["PCA2"], c="skyblue", s=8, label="Normal", alpha=0.5)

# 异常样本：红色 x 标记
plt.scatter(anomaly["PCA1"], anomaly["PCA2"], c="red", s=25, marker="x", label="Anomaly")

# 图形细节
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Visualization with Anomaly Labels (0 = normal, 1 = anomaly)")
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig("pca.png", dpi=300)
print("✅ 图像保存为 pca.png")
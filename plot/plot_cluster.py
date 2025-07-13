import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train_df = pd.read_csv("../build/train_clusters.csv")
test_df = pd.read_csv("../build/test_clusters.csv")

# 绘制每个簇的距离分布
plt.figure(figsize=(10, 5))
sns.boxplot(x='Cluster', y='Distance', data=train_df)
plt.title("Cluster-wise Distance Distribution (Train Set)")
plt.savefig("cluster_distance_train.png")
plt.show()

# 异常样本聚类分布
plt.figure(figsize=(10, 5))
sns.histplot(data=test_df[test_df["Label"] == 1], x="Cluster", bins=20, discrete=True)
plt.title("Abnormal Sample Cluster Assignment")
plt.savefig("abnormal_cluster_hist.png")
plt.show()

# 异常样本的距离散点图
plt.figure(figsize=(10, 5))
sns.scatterplot(x="Cluster", y="Distance", hue="Label", data=test_df, palette="Set1")
plt.title("Cluster Distance of Test Samples (Abnormal vs Normal)")
plt.savefig("test_distance_scatter.png")
plt.show()

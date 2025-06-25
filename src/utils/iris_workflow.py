# iris_workflow.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Step 1: 加载 Iris 数据
iris = load_iris()
X = iris.data        # shape: (150, 4)
y = iris.target      # shape: (150,)

# Step 2: 保存为 .npy 文件
np.save("iris_features.npy", X)
np.save("iris_labels.npy", y)

print("已保存 iris_features.npy 和 iris_labels.npy")

# Step 3: 读取 .npy 文件
X_loaded = np.load("iris_features.npy")
y_loaded = np.load("iris_labels.npy")

# Step 4: 可视化（方式一：前两个原始特征）
plt.figure(figsize=(8, 6))
plt.scatter(X_loaded[:, 0], X_loaded[:, 1], c=y_loaded, cmap='viridis', edgecolors='k')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset: Sepal Features')
plt.colorbar(ticks=[0, 1, 2], label='Species')
plt.grid(True)
plt.savefig("iris_sepal_scatter.png", dpi=300)
plt.show()

# Step 5: 可视化（方式二：PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_loaded)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_loaded, cmap='viridis', edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Iris Dataset: PCA Visualization')
plt.colorbar(ticks=[0, 1, 2], label='Species')
plt.grid(True)
plt.savefig("iris_pca_scatter.png", dpi=300)
plt.show()
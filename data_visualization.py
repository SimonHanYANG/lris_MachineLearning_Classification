import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets

# save path root
save_path_root = "./data_visualization/"
# check path root exist or not
if not os.path.exists(save_path_root):
    os.makedirs(save_path_root)

# load iris dataset
iris = datasets.load_iris()

# create DataFrame
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# create pairplot
'''
Pairplot

Pairplot 是 Seaborn 库中的一个功能，它可以创建一个图形矩阵来表示数据集中的每对特征之间的关系。在这个矩阵中，主对角线上的图是单个特征的直方图或密度估计。

对于非对角线上的图，横坐标和纵坐标分别代表两个不同的特征，每个点代表一个样本。你可以看到所有可能的特征对之间的散点图。

Pairplot 可以非常直观地显示数据集中各个特征之间的相关性，以及每个特征自身的分布情况。通过颜色标识不同的类别，我们还可以观察到不同类别在特征空间中的分布差异。
'''
pairplot_fig = sns.pairplot(iris_df, hue='target')
save_pairplot_path = save_path_root + "pairplot.png"
pairplot_fig.savefig(save_pairplot_path)  # save pairplot

# create a new matplotlib fig
plt.figure()

# create a scatter plot
'''
Scatterplot

Scatterplot 是一个二维图，显示了两个特征（在这个例子中是 'sepal length (cm)' 和 'sepal width (cm)'）之间的关系。每个点代表一个样本，横坐标代表 'sepal length (cm)'，纵坐标代表 'sepal width (cm)'。

与 Pairplot 中的非对角线图类似，Scatterplot 可以直观地显示两个特征之间的相关性，并通过颜色标识不同的类别，使我们能够观察到不同类别在这两个特征空间中的分布差异。
'''
scatterplot_fig = sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=iris_df)
save_scatterplot_path = save_path_root + "scatterplot.png"
plt.savefig(save_scatterplot_path)  # save scatter plot

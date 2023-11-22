import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Linear Regression
from sklearn.ensemble import RandomForestClassifier

from dataset import IRISDataset

# save path root
save_path_root = "./random_forest/"
# check path root exist or not
if not os.path.exists(save_path_root):
    os.makedirs(save_path_root)

# load dataset
X_train, X_test, y_train, y_test = IRISDataset()

# create model
model = RandomForestClassifier()
# train
model.fit(X_train, y_train)
# pred
y_pred = model.predict(X_test)

# calculate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n {conf_matrix}")

# Results ouput:
'''
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    Confusion Matrix:
    [[10  0  0]
    [ 0  9  0]
    [ 0  0 11]]
'''

# plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", square = True, cmap = 'Blues_r', ax = ax)
ax.set_ylabel('Actual label')
ax.set_xlabel('Predicted label')
ax.set_title('Confusion Matrix')

# save confusion matrix
save_conf_path = save_path_root + "confusion_matrix.png"
fig.savefig(save_conf_path)

'''
在混淆矩阵中，行表示实际类别，列表示预测类别。每个单元格的数字表示该类别的样本数量。
对角线上的单元格表示正确分类的样本数量，非对角线上的单元格表示被错误分类的样本数量。
'''

'''
accuracy、precision 和 recall 都是1.0，这说明你的模型在测试集上的表现非常好。具体来说：

准确率（Accuracy）：是所有预测正确的观测值（真阳性和真阴性）占总观测值的比率。准确率为1.0意味着所有的预测结果都是正确的，没有一个样本被错误分类。

精确度（Precision）：是所有真正类（真阳性）占所有预测为正类（真阳性和假阳性）的比率。精确度为1.0意味着所有你预测为正类的样本都是真正的正类，没有一次假阳性错误（即没有把负类错误地预测为正类）。

召回率（Recall）：也被称为真阳性率或者灵敏度，是所有真正类（真阳性）占所有实际为正类（真阳性和假阴性）的比率。召回率为1.0意味着你正确地预测出了所有的正类，没有一次假阴性错误（即没有漏掉任何正类）。
'''
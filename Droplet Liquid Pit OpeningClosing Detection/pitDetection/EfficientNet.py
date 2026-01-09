# =========================================SVM识别=======================================
import os

import matplotlib
import torch
from torchvision import models, transforms
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn import AdaptiveAvgPool2d, Flatten
from sklearn.metrics import roc_curve, auc
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from joblib import load, dump

# 图像转换流程，根据EfficientNet的需要，调整为更适合其输入的尺寸
transform = Compose([
    Resize((224, 224)),  # EfficientNet通常使用不同的输入尺寸，例如B0使用224, B7使用600等
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载预训练的EfficientNet模型
model = models.efficientnet_b0(pretrained=True)
model.classifier = Flatten()  # 修改模型的分类器部分以适应特征提取
model.eval()

# 保存模型权重
torch.save(model.state_dict(), 'efficientnet_features_extractor.pth')

# 使用示例输入张量以便于跟踪
example = torch.rand(1, 3, 224, 224)

# 转换模型为TorchScript
traced_script_module = torch.jit.trace(model, example)

# 保存转换的模型
traced_script_module.save('efficientnet_traced.pt')

# 函数：从DataFrame中加载图像及其标签
def load_images_and_labels(df_labels, transform):
    images = []
    labels = []
    for _, row in df_labels.iterrows():
        image_path = row['filename']
        label = row['label']
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)
        labels.append(label)
    return images, labels

# CSV文件路径和标签处理
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.abspath(os.path.join(CURRENT_DIR, 'DataSet/labels.csv'))
df_labels = pd.read_csv(csv_file_path)
root_dir = os.path.abspath(os.path.join(CURRENT_DIR, 'DataSet'))
# csv_file_path = 'C:\\Users\\ZQ\\train\\labels.csv'
# df_labels = pd.read_csv(csv_file_path)
# root_dir = 'C:\\Users\\ZQ\\train'
df_labels['filename'] = df_labels['filename'].apply(lambda x: os.path.join(root_dir, x))

# 加载图像和标签
loaded_images, labels = load_images_and_labels(df_labels, transform)
# 输出所有唯一的标签来检查是否只有两个类别
unique_labels = pd.unique(df_labels['label'])
print("Unique Labels in DataFrame:", unique_labels)

# 将加载的图像Tensor列表转换为批处理Tensor
image_tensors = torch.stack(loaded_images)

# 标签编码
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print("Encoded labels:", labels)
print("Classes in Label Encoder:", label_encoder.classes_)

# 特征提取函数
def extract_features(image_tensors):
    model.eval()  # 将模型设置为评估模式
    features = []
    with torch.no_grad():
        for data in image_tensors:
            feature = model(data.unsqueeze(0))  # 增加批次维度
            features.append(feature.squeeze().numpy())
    return features

# 提取特征
X_features = extract_features(image_tensors)

# 划分数据集

X_train_features, X_test_features, y_train, y_test = train_test_split(X_features, labels, test_size=0.4, random_state=42)
print("Labels in training set:", y_train)
print("Labels in test set:", y_test)

# 训练SVM模型
svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_features, y_train)
# 使用训练好的模型预测测试集的概率（注意获取正类的概率）
y_prob = svm_model.predict_proba(X_test_features)[:, 1]  # 获取属于正类的概率

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 计算 AUC
roc_auc = auc(fpr, tpr)

# 打印 AUC
print(f"AUC: {roc_auc:.4f}")
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'bold'  # 全局加粗
# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机猜测的对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FP Rate',fontsize=24)
plt.ylabel('TP Rate',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title('ROC Curve',fontsize=24)
plt.legend(loc='lower right',fontsize=22)
plt.show()

# 评估模型
y_pred = svm_model.predict(X_test_features)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,annot_kws={"size":50})
plt.xlabel('Predicted',fontsize=50)
plt.ylabel('True',fontsize=50)
# 调整刻度标识大小
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Confusion Matrix',fontsize=20)
plt.show()

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with deep features: {test_accuracy * 100:.2f}%")
# 计算F1分数
f1 = f1_score(y_test, y_pred, average='binary')  # If it's a binary classification task
print(f"F1 Score: {f1:.4f}")
# 计算和打印每个类别的概率
probas = svm_model.predict_proba(X_test_features)
average_probas = np.mean(probas, axis=0)
print("Class probabilities:")
for i, proba in enumerate(average_probas):
    print(f"{i}: {proba:.2f}")

# 保存SVM模型
dump(svm_model, 'svm_classifier_efficientnet.joblib')
# 加载SVM模型
svm_model = load('svm_classifier_efficientnet.joblib')
# 函数：从文件夹加载图像，无需标签
def load_images(folder_path, transform):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 假设图像为jpg或png格式
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
            filenames.append(filename)
    return images, filenames
# 生成CSV文件的函数
def predict_and_generate_csv(images, filenames, model, output_file='predictions3.csv'):
    image_tensors = torch.stack(images)  # 转换为批处理张量
    features = extract_features(image_tensors)  # 特征提取
    predictions = model.predict(features)  # 使用SVM模型进行预测
    results = pd.DataFrame({'Filename': filenames, 'Label': predictions})
    output_dir = r'C:\Users\付诗捷\Desktop\开源代码\开口闭合检测\数据集\test'
    full_output_path = os.path.join(output_dir, output_file)
    results.to_csv(full_output_path, index=False)
    print("Predictions saved to:", full_output_path)  # 正确显示完整路径

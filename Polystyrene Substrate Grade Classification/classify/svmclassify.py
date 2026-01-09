import os
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import models
from torch.nn import AdaptiveAvgPool2d, Flatten
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2


# ==================== 1. 修改数据路径 ====================
# 1. 获取当前脚本所在的目录 (base_dir)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 使用 os.path.join 拼接相对路径
# 对应逻辑: 当前目录 -> DataSet -> json -> train -> data
csv_file_path = os.path.join(base_dir, 'DataSet', 'train_ready', 'labels.csv')

# 对应逻辑: 当前目录 -> DataSet -> train_ready
root_dir = os.path.join(base_dir, 'DataSet', 'train_ready')

# 读取 CSV
df_labels = pd.read_csv(csv_file_path)

# 拼接完整路径：将 root_dir 和 CSV 里的文件名拼在一起
df_labels['filename'] = df_labels['filename'].apply(lambda x: os.path.join(root_dir, x))

# ========================================================

# 图像转换流程
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# ==================== 2. 模型为 ResNet50 ====================
print("正在加载 ResNet50 预训练模型...")
model = models.resnet50(pretrained=True)

model = torch.nn.Sequential(*(list(model.children())[:-1]),
                            AdaptiveAvgPool2d(output_size=(1, 1)),
                            Flatten())
model.eval()

# ==================== 3. 修改模型保存路径为当前目录 ====================
# 去掉绝对路径，直接写文件名，就会保存到脚本所在的目录
torch.save(model.state_dict(), "resnet50_features_extractor.pth")
print("模型权重已保存为 resnet50_features_extractor.pth")

# 使用示例输入张量以便于跟踪
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet50_traced.pt")
print("TorchScript 模型已保存为 resnet50_traced.pt")


# =================================================================

# 特征提取函数
def extract_features(image_tensors):
    model.eval()
    features = []
    with torch.no_grad():
        for data in image_tensors:
            feature = model(data.unsqueeze(0))
            features.append(feature.squeeze().numpy())
    return np.array(features)


def load_images_and_labels(df_labels, transform):
    images = []
    labels = []
    print("正在加载图像数据...")
    for idx, row in df_labels.iterrows():
        image_path = row['filename']
        label = row['label']
        try:
            # 增加异常处理，防止某张图坏了导致程序崩溃
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
            labels.append(label)
        except Exception as e:
            print(f"无法读取图像: {image_path}, 错误: {e}")

    return images, labels


def predict_image_class_proba(image_path, model, svm_model, label_map, transform):
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    # 提取特征
    features = extract_features(image_tensor)
    # 预测概率
    probas = svm_model.predict_proba(features)[0]
    # 使用正确的label_map和svm_model.classes_获取类别名称
    class_probas = {label_map[class_idx]: proba for class_idx, proba in zip(svm_model.classes_, probas)}
    return class_probas


# 加载数据和标签
loaded_images, labels = load_images_and_labels(df_labels, transform)

if len(loaded_images) == 0:
    print("错误：没有加载到任何图像，请检查路径配置！")
    exit()

# 转换为 Tensor
image_tensors = torch.stack(loaded_images)

# 标签编码
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print(f"类别映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 提取特征
print("开始提取图像特征...")
X_features = extract_features(image_tensors)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 训练 SVM 模型
print("开始训练 SVM 模型...")
svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 计算准确率
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
val_accuracy = accuracy_score(y_val, svm_model.predict(X_val))
test_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 保存 SVM 模型
dump(svm_model, 'svm_classifier.joblib')
print("SVM 模型已保存为 svm_classifier.joblib")

# ==================== 4. 自动测试一张图片 ====================

# 自动选取数据集中的第一张图片进行测试。
print("-" * 30)
if len(df_labels) > 0:
    test_image_path = df_labels.iloc[0]['filename']  # 取第一张图
    print(f"正在测试示例图片: {os.path.basename(test_image_path)}")

    # 构造反向映射 (从数字到类别名)
    inv_label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

    class_probas = predict_image_class_proba(test_image_path, model, svm_model, inv_label_map, transform)

    print("预测结果 (类别概率):")
    for label, proba in class_probas.items():
        print(f"类别 {label}: {proba:.2f}")
else:
    print("没有图片可供测试。")
print("-" * 30)
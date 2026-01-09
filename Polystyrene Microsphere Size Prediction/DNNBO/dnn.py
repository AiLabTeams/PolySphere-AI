import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# 设置 TensorFlow 的全局随机数种子
tf.random.set_seed(42)
np.random.seed(42)  # 设置 NumPy 的随机数种子，虽然对 TensorFlow 模型权重初始化不直接影响，但可能影响数据预处理

# 加载数据集
# 获取当前脚本所在的目录 (current_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接路径：当前目录 + DataSet + 文件名
file_path = os.path.join(current_dir, 'DataSet', 'bodata.csv')

# 读取数据
data = pd.read_csv(file_path)

# 假设 'feature1', 'feature2', 'feature3', 'feature4' 是数值特征列
# 'type' 是类别特征列（例如: 'type1', 'type2'）
# 'size' 是要预测的目标列
X = data[['KPS', 'SDS', 'H2O', 'C2H5OH', 'ST', 'type']].values
y_size = data['size'].values
y_cv = data['cv'].values

# 分类标签处理
y_cv_categorical = pd.cut(y_cv, bins=[-np.inf, 3.31, np.inf], labels=[0, 1])
y_cv_categorical = y_cv_categorical.astype(int)  # 将分类转换为整数编码
y_cv_one_hot = to_categorical(y_cv_categorical, num_classes=2)  # 转换为独热编码

# 分离数值特征和类别特征
X_numeric = X[:, :5]
X_categorical = X[:, 5]

# 对数值特征进行标准化
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# 将类别字符串转换为整数
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_categorical)

# 获取类别数量
n_types = len(label_encoder.classes_)

# 对整数编码的类别特征进行独热编码
X_categorical_encoded = to_categorical(integer_encoded, num_classes=n_types)

# 合并数值特征和编码后的类别特征
X_combined = np.concatenate((X_numeric_scaled, X_categorical_encoded), axis=1)

# 划分数据集为训练集和测试集
X_train, X_test, y_train_size, y_test_size = train_test_split(X_combined, y_size, test_size=0.2, random_state=42)
X_train, X_test, y_train_class_one_hot, y_test_class_one_hot = train_test_split(X_combined, y_cv_one_hot, test_size=0.2,
                                                                                random_state=42)


# 创建神经网络构建函数
def create_model():
    input_layer = Input(shape=(X_combined.shape[1],))
    dense1 = Dense(10, activation='relu')(input_layer)
    dense2 = Dense(100, activation='relu')(dense1)
    dense3 = Dense(100, activation='relu')(dense2)
    dense4 = Dense(10, activation='relu')(dense3)
    output_regression = Dense(1, name='output_regression', activation='linear')(dense4)
    output_classification = Dense(2, name='output_classification', activation='softmax')(dense4)
    model = Model(inputs=input_layer, outputs=[output_regression, output_classification])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss={'output_regression': 'mean_squared_error', 'output_classification': 'categorical_crossentropy'},
                  loss_weights={'output_regression': 0.5, 'output_classification': 0.5},  # 分配损失权重
                  metrics={'output_regression': ['mae'], 'output_classification': ['accuracy']})
    return model


model = create_model()

# 训练模型
history = model.fit(X_train, [y_train_size, y_train_class_one_hot], epochs=150, batch_size=1,
                    validation_split=0.1, verbose=1)

# 评估模型
loss = model.evaluate(X_test, [y_test_size, y_test_class_one_hot], verbose=0)
print('Test loss:', loss)

# 使用模型进行预测
predictions = model.predict(X_test)

# 输出预测结果和实际结果
print('Predictions:', predictions)
print('Actual results size:', y_test_size)
print('Actual results classification:', y_test_class_one_hot)

# 计算回归指标
mse = mean_squared_error(y_test_size, predictions[0])
r2 = r2_score(y_test_size, predictions[0])
mae = mean_absolute_error(y_test_size, predictions[0])
accuracy = accuracy_score(y_test_class_one_hot.argmax(axis=1), predictions[1].argmax(axis=1))

# 输出回归指标
print('Mean Squared Error (MSE):', mse)
print('R^2 Score', r2)
print('Mean Absolute Error (MAE):', mae)
print('Classification accuracy:', accuracy)

# 可视化训练过程
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# 回归任务的可视化
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.scatter(y_test_size, predictions[0], color='blue', label='Predictions')
plt.scatter(y_test_size, y_test_size, color='red', label='Actual')
plt.title('Regression Task - Predictions vs Actual')
plt.xlabel('Actual size')
plt.ylabel('Predicted size')
plt.legend()

# 分类任务的可视化 - 使用混淆矩阵
predicted_classes = predictions[1].argmax(axis=1)
actual_classes = y_test_class_one_hot.argmax(axis=1)
class_names = ['Class 0', 'Class 1', 'Class 2']

# 计算混淆矩阵
cm = confusion_matrix(actual_classes, predicted_classes)

# 使用seaborn绘制混淆矩阵
plt.subplot(2, 1, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix - Classification Task')
plt.show()
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
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
X = data[['KPS', 'SDS', 'H2O', 'C2H5OH', 'ST', 'type']].values
y_size = data['size'].values
y_cv = data['cv'].values

# 分类标签
y_classification = (y_cv > 3.1).astype(int)  # 假设cv>3.1时为1，否则为0

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

# K折交叉验证
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 存储每次迭代的损失和分数
fold_losses = []
fold_r2_scores = []
fold_mean_abs_errors = []
fold_accuracies = []


# 创建神经网络构建函数
def create_model():
    input_layer = Input(shape=(X_combined.shape[1],))
    dense1 = Dense(10, activation='relu')(input_layer)
    dense2 = Dense(100, activation='relu')(dense1)
    dense3 = Dense(100, activation='relu')(dense2)
    dense4 = Dense(10, activation='relu')(dense3)
    output_regression = Dense(1, name='output_regression', activation='linear')(dense4)
    output_classification = Dense(1, name='output_classification', activation='sigmoid')(dense4)
    model = Model(inputs=input_layer, outputs=[output_regression, output_classification])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss={'output_regression': 'mean_squared_error', 'output_classification': 'binary_crossentropy'},
                  loss_weights={'output_regression': 0.5, 'output_classification': 0.5},  # 分配损失权重
                  metrics={'output_regression': ['mae'], 'output_classification': ['accuracy']})
    return model

# 对每个折进行迭代
for fold, (train_index, val_index) in enumerate(kf.split(X_combined, y_size)):
    print(f'Fold {fold + 1}')
    X_train_fold, X_val_fold = X_combined[train_index], X_combined[val_index]
    Y_train_fold, Y_val_fold = y_size[train_index], y_size[val_index]
    Y_class_train_fold, Y_class_val_fold = y_classification[train_index], y_classification[val_index]

    model = create_model()
    history = model.fit(X_train_fold, [Y_train_fold, Y_class_train_fold], epochs=150, batch_size=1, verbose=0)
    val_loss = model.evaluate(X_val_fold, [Y_val_fold, Y_class_val_fold], verbose=0)
    fold_losses.append(val_loss[0])
    val_predictions = model.predict(X_val_fold)
    abe = mean_absolute_error(Y_val_fold, val_predictions[0])
    fold_mean_abs_errors.append(abe)
    r2 = r2_score(Y_val_fold, val_predictions[0])
    fold_r2_scores.append(r2)
    accuracy = accuracy_score(Y_class_val_fold, (val_predictions[1] > 0.5).astype(int))
    fold_accuracies.append(accuracy)

    print(f'Validation loss: {val_loss[0]}, Regression MAE: {val_loss[2]}, Classification Accuracy: {val_loss[3]}')
    print('R2 score:', r2)
    print('mean_absolute_error:', abe)
    print('Classification accuracy:', accuracy)

# 输出所有折的平均损失和分数
avg_loss = np.mean(fold_losses)
avg_r2_score = np.mean(fold_r2_scores)
avg_error = np.mean(fold_mean_abs_errors)
avg_accuracy = np.mean(fold_accuracies)
print(f'Average validation loss: {avg_loss}')
print(f'Average R2 score: {avg_r2_score}')
print(f'Average error: {avg_error}')
print(f'Average classification accuracy: {avg_accuracy}')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
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
history = model.fit(X_combined, [y_size, y_cv_one_hot], epochs=150, batch_size=1, validation_split=0,
                    verbose=2)

# 保存模型
model_save_path = 'my_model.h5'  # 定义模型保存的路径和文件名
model.save(model_save_path)

while True:
    try:
        # 用户输入数据
        input_data = input(
            "请输入预测数据，特征用逗号分隔 ('KPS,SDS,H2O,C2H5OH,ST,type'), 输入'quit'退出: ")

        if input_data.lower() == 'quit':
            break

        # 分割输入数据
        input_data = input_data.split(',')

        # 输入数据格式化处理
        input_numeric = np.array(input_data[:5], dtype=np.float64)  # 将数值特征转为float64类型

        # 对数值特征进行标准化
        input_numeric_scaled = scaler.transform([input_numeric])  # 注意[ ]表示单个样本

        # 对其加入type量特征
        st_type = input_data[5]  # 第六个数据为type
        int_encoded = label_encoder.transform([st_type])  # 注意[ ]表示单个样本
        input_type_one_hot = to_categorical(int_encoded, num_classes=n_types)  # 转换为独热编码
        input_combined = np.concatenate((input_numeric_scaled, input_type_one_hot), axis=1)

        # 使用模型进行预测
        prediction = model.predict(input_combined)
        print("预测结果：", prediction)  # 输出预测结果

    except Exception as e:
        print("发生错误，请检查您的输入是否正确（数值是否合理、类型是否存在）")
        print("错误详情：", e)

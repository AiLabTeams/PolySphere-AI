import os
import json
import cv2
import numpy as np
import pandas as pd
import glob

# ================= 配置区域 =================
# 1. 获取当前脚本所在的目录 (base_dir)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 使用 os.path.join 拼接相对路径
# 对应逻辑: 当前目录 -> DataSet -> json -> train -> data
input_folder = os.path.join(base_dir, 'DataSet', 'data')

# 对应逻辑: 当前目录 -> DataSet -> train_ready
output_folder = os.path.join(base_dir, 'DataSet', 'train_ready')


# ===========================================

def cv_imread(file_path):
    """
    专门解决 OpenCV 无法读取中文路径图片的问题
    """
    try:
        # 使用 numpy 读取文件流，然后用 opencv 解码
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, -1)  # -1 表示按原样读取（保留通道）
        return img
    except Exception as e:
        print(f"读取图片异常: {e}")
        return None


def cv_imwrite(file_path, img):
    """
    专门解决 OpenCV 无法保存图片到中文路径的问题
    """
    try:
        # 获取文件后缀
        ext = os.path.splitext(file_path)[1]
        # 编码图片数据
        retval, buf = cv2.imencode(ext, img)
        if retval:
            # 保存到文件
            buf.tofile(file_path)
            return True
    except Exception as e:
        print(f"保存图片异常: {e}")
    return False


def convert_json_to_dataset():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出目录: {output_folder}")

    data_list = []

    # 获取所有 json 文件
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    print(f"输入路径: {input_folder}")
    print(f"找到 {len(json_files)} 个 JSON 文件。开始处理...")

    count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # 尝试匹配 jpg 或 png
            base_name = os.path.splitext(os.path.basename(json_file))[0]

            # 优先尝试同名的 jpg
            image_path = os.path.join(input_folder, base_name + '.jpg')
            if not os.path.exists(image_path):
                # 如果 jpg 不存在，尝试 png
                image_path = os.path.join(input_folder, base_name + '.png')

            if not os.path.exists(image_path):
                print(f"[警告] 找不到对应的图片文件 (jpg/png): {base_name}")
                continue

            # === 修改点：使用自定义的中文读取函数 ===
            img = cv_imread(image_path)
            # ====================================

            if img is None:
                print(f"[错误] 图片读取失败 (可能是损坏): {image_path}")
                continue

            # 如果是 png 图片可能是 4 通道的，转为 3 通道 RGB
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            for i, shape in enumerate(content['shapes']):
                label = shape['label']
                points = shape['points']
                points_np = np.array(points, dtype=np.int32)

                x, y, w, h = cv2.boundingRect(points_np)

                # 边界保护
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)

                if w <= 0 or h <= 0:
                    continue

                crop_img = img[y:y + h, x:x + w]

                new_filename = f"{base_name}_{label}_{i}.jpg"
                save_path = os.path.join(output_folder, new_filename)

                # === 修改点：使用自定义的中文保存函数 ===
                cv_imwrite(save_path, crop_img)
                # ====================================

                data_list.append({
                    'filename': new_filename,
                    'label': label
                })
                count += 1

        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        csv_save_path = os.path.join(output_folder, 'labels.csv')
        df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 防止 Excel 打开乱码
        print("=" * 30)
        print(f"处理完成！")
        print(f"共生成 {count} 张小图。")
        print(f"CSV 文件已保存在: {csv_save_path}")
        print("=" * 30)
    else:
        print("未找到任何有效的数据，请检查路径。")


if __name__ == '__main__':

    convert_json_to_dataset()

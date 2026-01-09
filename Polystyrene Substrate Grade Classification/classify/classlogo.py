
#######################################################################
import os
import cv2
import numpy as np
from PIL import Image
from numpy import median
from sklearn.cluster import KMeans
import sys
import torch
from torchvision import transforms, models
from torchvision.transforms import Resize, ToTensor, Normalize
from joblib import load
from torch.nn import AdaptiveAvgPool2d, Flatten
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

sys.path.append("../..")
from segment_anything import sam_model_registry, SamPredictor
from statistics import mode
# 加载分类模型

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '/svm_model.joblib')

# 加载模型
svm_model = load(model_path)

# 图像预处理流程
transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载预训练的ResNet50模型并修改
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]),  # 移除最后的全连接层
                            AdaptiveAvgPool2d(output_size=(1, 1)),
                            Flatten())
model.eval()
def resize_image(image, target_size=(630, 480)):
    """ Resize the image and return the resized image and its corresponding scale """
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image, original_size

def extract_features(image_tensor):
    """ 使用预训练的模型提取特征 """
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0))  # 增加批次维度
    return features.view(-1).numpy()

def segment_image_kmeans(image_path, n_clusters=4):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image, _ = resize_image(image)
    pixels = resized_image.reshape((-1, 3))
    pixels = np.float32(pixels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(pixels)
    segmented_image = labels.reshape(resized_image.shape[:2])
    label_colors = kmeans.cluster_centers_.astype("uint8")
    segmented_display = label_colors[labels].reshape(resized_image.shape)
    return resized_image, segmented_image, segmented_display, labels, label_colors

def create_color_masks(segmented_display, colors):
    masks = {}
    for color, name in colors:
        lower = np.array([max(c - 10, 0) for c in color])
        upper = np.array([min(c + 10, 255) for c in color])
        mask = cv2.inRange(segmented_display, lower, upper)
        masks[name] = mask
    return masks

def load_and_prepare_model(device='cpu'):
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    return predictor

def select_subregions_within_mask(image, mask, region_index, num_regions=3, region_size=(16, 16), output_dir='subregions', transform=None, svm_model=None, extract_features=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subregions = []
    predictions = []  # This will store class labels like 'class_4', etc.
    coords = np.where(mask == 255)
    if coords[0].size == 0:
        print("No valid areas found in the mask.")
        return subregions, []  # Returns empty lists if no areas

    zip_coords = list(zip(coords[0], coords[1]))
    np.random.shuffle(zip_coords)

    custom_index = 0
    class_labels = ['class_4', 'class_3', 'class_2', 'class_1']  # This assumes class indices are 0-3 respectively
    for x, y in zip_coords:
        if custom_index >= num_regions:
            break
        if x + region_size[0] <= mask.shape[0] and y + region_size[1] <= mask.shape[1]:
            if np.all(mask[x:x + region_size[0], y:y + region_size[1]] == 255):
                subregion = image[x:x + region_size[0], y:y + region_size[1]]
                subregion_image = Image.fromarray(subregion).convert('RGB')
                patch_filename = os.path.join(output_dir, f'subregion_{region_index}_{custom_index}.png')
                subregion_image.save(patch_filename)

                # patch_image = Image.open(patch_filename).convert('RGB')
                patch_tensor = transform(subregion_image)
                features = extract_features(patch_tensor)
                probabilities = svm_model.predict_proba([features])[0]
                predicted_class_index = np.argmax(probabilities)
                predicted_class = class_labels[predicted_class_index]  # Map index to label
                predictions.append(predicted_class)  # Store the class label instead of index
                subregions.append((predicted_class, x, y, x + region_size[0], y + region_size[1]))
                # 在图像上标注子区域的分类结果
                center_x = (x + (x + region_size[0])) // 2
                center_y = (y + (y + region_size[1])) // 2
                cv2.putText(image, str(predicted_class), (center_y, center_x),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                custom_index += 1
                # custom_index += 1


    return subregions, predictions


def assign_colors_to_classes(predictions):
    unique_classes = sorted(set(filter(None, predictions)))  # Ensure consistent order, filter out None values
    color_map = {cls: color for cls, color in zip(unique_classes, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])}
    return color_map

def apply_color_to_masks(masks, predictions):
    combined_mask = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
    color_map = assign_colors_to_classes(predictions)

    for mask, pred in zip(masks, predictions):
        if pred is not None:
            color = color_map[pred]
            combined_mask[mask > 0] = color

    return combined_mask

def merge_masks_by_class(masks, median_classes):
    if not masks or not median_classes:  # 检查输入有效性
        return np.zeros((1, 1), dtype=np.uint8)

    height, width = masks[0].shape
    merged_mask = np.zeros((height, width), dtype=np.uint8)
    class_label = 1

    for i, median_class_i in enumerate(median_classes):
        if median_class_i is None:
            continue  # 跳过无效的类别
        mask_i = masks[i] > 0
        for j, median_class_j in enumerate(median_classes):
            if median_class_i == median_class_j:
                mask_j = masks[j] > 0
                merged_mask[mask_j] = class_label  # 此处确保不重复设置同一像素
        class_label += 1

    return merged_mask

def create_difference_masks(masks):
    previous_mask = None
    difference_masks = []
    for mask in masks:
        current_mask = (mask > 0).astype(np.uint8) * 255
        if previous_mask is not None:
            difference = cv2.absdiff(previous_mask, current_mask)
            difference_masks.append(difference)
        previous_mask = current_mask
    # 添加最后一个 current_mask 到列表中，以便也对其进行处理
    if current_mask is not None:
        difference_masks.append(current_mask)  # 确保也包括最后一个掩码
    return difference_masks
# 计算 IoU 和 Dice 系数
def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    dice = 2.0 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(true_mask))
    return dice
def main():
    #待处理的原始输入图像路径
    image_path = None

    #这张图片对应的真实标签，人工标注好的黑白掩码图
    true_mask_path = None  # 真实标签的路径

    original_image, segmented_image, segmented_display, labels, cluster_centers = segment_image_kmeans(image_path)
    predictor = load_and_prepare_model()
    resized_image,original_size = resize_image(original_image)
    predictor.set_image(resized_image)
    input_point = np.array([[280, 240]])
    input_label = np.array([1])

    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        difference_masks = create_difference_masks(masks)  # 确保掩码被初始化
        # 读取真实标签掩码
        # Resize masks to match the original image size (if necessary)
        # resized_masks = [cv2.resize(mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST) for
        #                  mask in masks]

        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask is None:
            raise ValueError(f"Failed to load true mask from {true_mask_path}")
            # Resize true_mask to match the resized image
        true_mask = cv2.resize(true_mask, (resized_image.shape[1], resized_image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        # 初始化 IoU 和 Dice 系数的总和
        total_iou = 0
        total_dice = 0

        # all_predictions = []
        median_classes = []

        median_classes = []  # 存储每个掩码的中位数类别
        plt.figure(figsize=(10, 10))  # Figure 1
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # 显示原始图像
        for i, diff_mask in enumerate(difference_masks):
            pred_mask = diff_mask > 0  # 将掩码二值化

            iou = calculate_iou(pred_mask, true_mask)
            dice = calculate_dice(pred_mask, true_mask)

            print(f"Mask {i + 1} - IoU: {iou:.4f}")

            total_iou += iou
            total_dice += dice

            # 计算平均 IoU 和 Dice 系数
        avg_iou = total_iou / len(difference_masks)
        avg_dice = total_dice / len(difference_masks)
        print(f"Average IoU: {avg_iou:.4f}")
        # print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")

        subregions, predictions = select_subregions_within_mask(original_image, diff_mask, i, num_regions=3,
                                                                region_size=(64, 64), output_dir='../saved_subregions',
                                                                transform=transform, svm_model=svm_model,
                                                                extract_features=extract_features)
        # 显示带标注的图像
        cv2.imshow("Annotated Image", original_image)
        if predictions:
            try:
                most_common_class = mode(predictions)  # Directly get the most common class
                print(f"Median class for mask {i + 1}: {most_common_class}")
                median_classes.append(most_common_class)  # Append the result directly
            except Exception as e:
                print(f"Error calculating mode for mask {i + 1}: {e}")
                median_classes.append(None)
        else:
            median_classes.append(None)  # 如果没有预测，则添加None
            # coords = np.where(diff_mask)
        if subregions:
            subregions.sort(key=lambda x: x[1])
            median_subregion = subregions[len(subregions) // 2]
            center_x = (median_subregion[1] + median_subregion[3]) // 2
            center_y = (median_subregion[2] + median_subregion[4]) // 2
            cv2.putText(original_image, str( most_common_class), (center_y, center_x), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)
        previous_mask = None
        combined_mask = np.zeros_like(original_image)  # For accumulating mask differences
        colors = [([255, 0, 0], 'Red'), ([0, 255, 0], 'Green'), ([0, 0, 255], 'Blue')]

        color_masks = create_color_masks(segmented_display, colors)

        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        for i, mask in enumerate(masks):
            current_mask = (mask > 0).astype(np.uint8) * 255
            if previous_mask is not None:
                difference = cv2.subtract(previous_mask, current_mask)
                combined_mask[difference > 0] = np.array(colors[i % len(colors)][0])
            previous_mask = current_mask

        combined_mask[current_mask > 0] = [255, 255, 0]  # Yellow for the last mask

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.addWeighted(original_image, 0.7, combined_mask, 0.3, 0))
        plt.title("Combined Mask Differences")
        plt.axis('off')
        plt.show()
        cv2.imshow("Annotated Image", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Subregions with Class Labels")
        plt.axis('off')
        plt.show()

        if median_classes:
                # 这一行可能是问题所在，应该使用 median_classes 而不是 all_predictions
                merged_mask = merge_masks_by_class(difference_masks, median_classes)
                combined_mask = apply_color_to_masks(difference_masks, median_classes)
                final_image = cv2.addWeighted(original_image, 0.7, combined_mask, 0.3, 0)
                plt.figure(figsize=(10, 10))  # Figure 2
                plt.imshow(final_image)
                plt.title("Mask Differences")
                plt.axis('off')
                plt.show()

                # display_color_mapped_masks(original_image, merged_mask)
        else:
                print("No valid masks or predictions available for merging.")
                # Visualize the final merged mask
        plt.figure(figsize=(10, 10))
        colored_mask = cv2.applyColorMap(merged_mask,
                                         cv2.COLORMAP_JET)  # Color map the merged mask for better distinction
        final_merged_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        plt.imshow(final_merged_image)
        plt.title("Merged Class Masks")
        plt.axis('off')
        plt.show()



    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

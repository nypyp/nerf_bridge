# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import numpy as np
import os
import time
import math
from pathlib import Path
import torch
import torch_scatter
from mmdeploy_runtime import Segmentor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='语义分割推理演示')
    parser.add_argument('device_name', help='设备名称, cuda或cpu')
    parser.add_argument('model_path', help='mmdeploy SDK模型路径')
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--mapping_file', default='ade20k_to_scannet_v2.csv', help='类别映射文件路径')
    return parser.parse_args()


def get_palette(num_classes=150):
    """生成随机调色板"""
    np.random.seed(42)
    return [tuple(c) for c in np.random.randint(0, 256, size=(num_classes, 3))]


def load_mapping(mapping_file):
    """
    从CSV文件加载类别映射关系
    
    参数:
        mapping_file (str): CSV文件路径
    
    返回:
        tuple: (mapping_list, num_scannet_classes)
    """
    mapping_list = []
    max_class = 0

    with open(mapping_file, 'r') as f:
        for line in f:
            _, target_class = line.strip().split(',')
            target_class = int(target_class)
            mapping_list.append(target_class if target_class != -1 else 0)  # -1映射到背景类
            max_class = max(max_class, target_class)

    return mapping_list, max_class + 1


def preprocess_output(seg):
    """
    预处理分割模型输出，确保为有效的概率分布张量
    
    参数:
        seg (np.ndarray): 分割器输出的概率分布数组，要求为3维数组，格式为(C,H,W)或(H,W,C)
    
    返回:
        torch.Tensor: 处理后的概率分布张量，格式为(C,H,W)
    """
    if not isinstance(seg, np.ndarray) or seg.dtype != np.float32 or len(seg.shape) != 3:
        raise ValueError("分割器输出必须是float32类型的3维numpy数组, dtype: ", seg.dtype," shape: ", seg.shape)

    # 转换为(C,H,W)格式
    probs = torch.from_numpy(seg if seg.shape[0] < seg.shape[1] else seg.transpose(2, 0, 1))

    # 确保是概率分布
    if not (0 <= probs.min() <= 1 and 0 <= probs.max() <= 1):
        probs = torch.nn.functional.softmax(probs, dim=0)

    if not torch.allclose(probs.sum(dim=0), torch.ones_like(probs[0]), atol=1e-3):
        raise ValueError("每个像素位置的概率之和必须接近1")
    
    return probs


def apply_semantic_mapping(probs, mapping_list, num_scannet_classes):
    """将源数据集的概率预测映射到ScanNet类别"""
    map_tensor = torch.tensor(mapping_list, device=probs.device).long().view(-1, 1, 1)
    scannet_probs = torch.zeros(num_scannet_classes, *probs.shape[1:], device=probs.device)
    
    # 使用scatter_add进行概率重映射
    torch_scatter.scatter_add(src=probs, index=map_tensor.expand_as(probs), dim=0, out=scannet_probs)
    
    # 重新归一化
    sum_probs = scannet_probs.sum(dim=0, keepdim=True)
    valid_mask = sum_probs > 0
    scannet_probs[:, valid_mask[0]] /= sum_probs[:, valid_mask[0]]
    
    return scannet_probs


def process_image(img_path, segmentor, palette, output_dir, device, mapping_list, num_scannet_classes):
    """处理单张图片"""
    img_read_start = time.time()
    img = cv2.imread(str(img_path))
    img_read_end = time.time()
    
    if img is None:
        print(f"无法读取图片: {img_path}")
        return False

    # 模型推理
    inference_start = time.time()
    seg = segmentor(img)
    probs = preprocess_output(seg).to(device)
    inference_end = time.time()
    

    # 应用语义映射
    mapping_start = time.time()
    final_probs = apply_semantic_mapping(probs, mapping_list, num_scannet_classes).cpu()
    mapping_end = time.time()

    # 可视化结果
    visualization_start = time.time()
    final_seg = torch.argmax(final_probs, dim=0).numpy()
    vis_img = visualize_segmentation(final_seg, img, palette)
    uncertainty_vis = visualize_uncertainty(final_probs, img)
    visualization_end = time.time()

    # 保存结果
    save_start = time.time()
    img_name = Path(img_path).stem + ".png"
    cv2.imwrite(str(output_dir / "seg" / img_name), vis_img)
    cv2.imwrite(str(output_dir / "uncertainty" / img_name), uncertainty_vis)
    save_end = time.time()

    # 打印用时
    print(f"图片读取: {img_read_end - img_read_start:.4f}s, "
          f"模型推理: {inference_end - inference_start:.4f}s, "
          f"语义映射: {mapping_end - mapping_start:.4f}s, "
          f"可视化: {visualization_end - visualization_start:.4f}s, "
          f"保存: {save_end - save_start:.4f}s")

    return True


def visualize_segmentation(seg, img, palette):
    """可视化分割结果"""
    color_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label] = color
    return cv2.addWeighted(img, 0.5, color_seg[..., ::-1], 0.5, 0)


def visualize_uncertainty(probs, img):
    """可视化不确定性（熵）"""
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=0) / math.log2(probs.shape[0])
    entropy_vis = (1 - entropy.numpy()) * 255
    entropy_vis = cv2.applyColorMap(entropy_vis.astype(np.uint8), cv2.COLORMAP_JET)
    return np.hstack([img, entropy_vis])


def main():
    args = parse_args()
    device = torch.device(args.device_name)

    # 初始化分割器和调色板
    segmentor = Segmentor(model_path=args.model_path, device_name=args.device_name, device_id=0)
    palette = get_palette()

    # 加载类别映射
    mapping_file = Path(args.mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"映射文件未找到: {mapping_file}")
    mapping_list, num_scannet_classes = load_mapping(mapping_file)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    (output_dir / "seg").mkdir(parents=True, exist_ok=True)
    (output_dir / "uncertainty").mkdir(parents=True, exist_ok=True)

    # 处理输入路径
    input_path = Path(args.image_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    processed = failed = 0
    for img_path in input_path.iterdir() if input_path.is_dir() else [input_path]:
        if img_path.suffix.lower() in image_extensions:
            if process_image(img_path, segmentor, palette, output_dir, device, mapping_list, num_scannet_classes):
                processed += 1
            else:
                failed += 1

    print(f"\n处理完成总结:")
    print(f"成功处理: {processed} 张图片")
    print(f"处理失败: {failed} 张图片")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
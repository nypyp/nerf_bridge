import argparse
import cv2
import numpy as np
from pathlib import Path
from mmdeploy_runtime import Segmentor
from tabulate import tabulate  # 需要安装tabulate库

def parse_args():
    parser = argparse.ArgumentParser(description='高效语义分割推理')
    parser.add_argument('device', help='设备名称（cuda/cpu）')
    parser.add_argument('model_path', help='模型路径')
    parser.add_argument('image_path', help='输入图片/目录路径')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--mapping', default='ade20k_to_scannet_v2.csv', help='映射文件路径')
    return parser.parse_args()

def format_list(lst, max_len=20):
    """格式化长列表显示"""
    if len(lst) > max_len:
        return f"[{', '.join(map(str, lst[:max_len]))}, ...]"
    return str(lst)

def load_mapping(mapping_file):
    mapping = []
    with open(mapping_file, 'r') as f:
        for line in f:
            _, target = line.strip().split(',')
            mapping.append(int(target) if int(target)!=-1 else 0)
    return mapping

def get_palette(num_classes=150):
    np.random.seed(42)
    return [tuple(np.random.randint(0,255,3)) for _ in range(num_classes)]

def visualize_segmentation(seg, img, palette):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label] = color
    return cv2.addWeighted(img, 0.5, color_seg[...,::-1], 0.5, 0)

def process_image(img_path, segmentor, mapping, palette, output_dir):
    img = cv2.imread(str(img_path))
    if img is None: return False, None, None

    # 直接获取mask输出
    seg = segmentor(img)  # 假设mask是第一个输出元素
    
    # 应用类别映射
    final_seg = np.vectorize(lambda x: mapping[x])(seg)
    
    unique_before = np.unique(seg).tolist()
    unique_after = np.unique(final_seg).tolist()

    # 保存可视化结果
    vis_img = visualize_segmentation(final_seg, img, palette)
    output_path = output_dir / f"seg/{img_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)
    return True, unique_before, unique_after

def main():
    args = parse_args()
    segmentor = Segmentor(model_path=args.model_path, device_name=args.device, device_id=0)
    
    # 初始化映射和调色板
    mapping = load_mapping(args.mapping)
    palette = get_palette(len(mapping))
    
    # 创建输出目录
    output_dir = Path(args.output)
    (output_dir / "seg").mkdir(parents=True, exist_ok=True)

    # 处理输入
    input_path = Path(args.image_path)
    for img_file in input_path.iterdir() if input_path.is_dir() else [input_path]:
        if img_file.suffix.lower() in ('.jpg','.jpeg','.png'):
            # 修改调用方式，接收返回的类别信息
            success, before, after = process_image(img_file, segmentor, mapping, palette, output_dir)
            if success:
                print(f"Processed: {img_file} ")
                # print(tabulate([
                #     ['原类别', format_list(before)],
                #     ['映射后', format_list(after)]
                # ], headers=['类别类型', '内容'], tablefmt='psql'))

if __name__ == '__main__':
    main()

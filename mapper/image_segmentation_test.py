import time
import torch 
import cv2
from mmdeploy_runtime import Segmentor
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test inference time.')
    parser.add_argument('device_name', default='cuda', help='Device name.')
    parser.add_argument('model_path', help='Path to the model.')
    parser.add_argument('image_path', help='Path to the image.')
    parser.add_argument('--warm_up', type=int, default=10, help='Number of warm-up iterations.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of test iterations.')
    return parser.parse_args()

def test_inference_time(segmentor, img, warm_up=10, iterations=100):
    # Warm-up phase
    for _ in range(warm_up):
        _ = segmentor(img)  # 预热推理
        torch.cuda.synchronize()  # 确保 GPU 操作完成

    # Test phase
    start_time = time.time()
    for _ in range(iterations):
        _ = segmentor(img)  # 正式推理
        torch.cuda.synchronize()  # 确保 GPU 操作完成
    end_time = time.time()

    # 计算平均推理时间
    total_time = end_time - start_time
    avg_time = total_time / iterations
    return avg_time

if __name__ == '__main__':
    args = parse_args()
    segmentor = Segmentor(model_path=args.model_path, device_name=args.device_name, device_id=0)
    img = cv2.imread(args.image_path)

    if img is None:
        raise ValueError(f"无法读取图片: {args.image_path}")

    avg_inference_time = test_inference_time(
        segmentor, img, warm_up=args.warm_up, iterations=args.iterations
    )
    print(f"平均推理时间: {avg_inference_time:.4f} 秒")
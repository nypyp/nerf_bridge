import cv2
import numpy as np
import albumentations as A
import torch
from typing import List, Callable
from functools import partial
from mmdeploy_runtime import Segmentor

class TTASegmentor:
    def __init__(self, 
        base_segmentor: Callable,
        threshold: float = 0.5,
        device: str = "cuda"):
        """
        完整增强版TTA分割器
        
        :param base_segmentor: 返回[C, H, W] logits的基础分割模型
        :param num_classes: 分割类别数
        :param threshold: 二值分割阈值
        :param device: 计算设备
        """
        self.segmentor = base_segmentor
        self.threshold = threshold
        self.device = torch.device(device)
        self.augmentations = self._get_full_augmentations()

    def _get_full_augmentations(self) -> List[A.BasicTransform]:
        """完整增强配置（来自原始demo.py）"""
        base_aug = [
            A.HorizontalFlip(always_apply=True),
            A.RGBShift(always_apply=True),
            A.CLAHE(always_apply=True),
            A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
            A.RandomBrightnessContrast(always_apply=True),
            A.MedianBlur(blur_limit=7, always_apply=True),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), always_apply=True)
        ]
        
        # 组合增强（与原始代码完全一致）
        composed_aug = [
            A.Compose([base_aug[1], base_aug[2]]),  # RGBShift + CLAHE
            A.Compose([base_aug[2], base_aug[3]]),  # CLAHE + RandomGamma
            A.Compose([base_aug[1], base_aug[3]]),  # RGBShift + RandomGamma
            A.Compose([base_aug[2], base_aug[4]]),  # CLAHE + BrightnessContrast
            A.Compose([base_aug[5], base_aug[6]])   # MedianBlur + Sharpen
        ]
        
        return base_aug + composed_aug

    def _apply_augmentations(self, image: np.ndarray) -> List[tuple]:
        """应用完整增强并生成逆变换信息"""
        results = []
        
        # 原始图像 (无增强)
        results.append({
            "image": image,
            "inverse": lambda x: x,
            "is_flipped": False
        })
        
        # 处理每个增强
        for aug in self.augmentations:
            transformed = aug(image=image)
            results.append(self._process_augmentation(aug, transformed["image"]))
            
        return results

    def _process_augmentation(self, aug: A.BasicTransform, img: np.ndarray) -> dict:
        """处理单个增强的逆变换逻辑"""
        # 水平翻转的特殊处理
        if isinstance(aug, A.HorizontalFlip):
            return {
                "image": img,
                "inverse": partial(self._flip_inverse, dims=[-1]),  # 在W维度翻转
                "is_flipped": True
            }
        
        # 处理组合增强
        if isinstance(aug, A.Compose):
            return self._process_compose_aug(aug, img)
            
        # 默认处理（无空间变换）
        return {
            "image": img,
            "inverse": lambda x: x,
            "is_flipped": False
        }

    def _process_compose_aug(self, aug: A.Compose, img: np.ndarray) -> dict:
        """处理组合增强的逆变换链"""
        inverses = []
        is_flipped = False
        
        # 逆向处理每个子增强
        for t in reversed(aug.transforms):
            if isinstance(t, A.HorizontalFlip):
                inverses.append(partial(self._flip_inverse, dims=[-1]))
                is_flipped = True
            else:
                inverses.append(lambda x: x)
                
        return {
            "image": img,
            "inverse": lambda x: self._apply_chain_inverse(x, inverses),
            "is_flipped": is_flipped
        }

    def _flip_inverse(self, x: torch.Tensor, dims: list) -> torch.Tensor:
        """水平翻转逆操作（保持与原始代码一致）"""
        return torch.flip(x, dims=dims)

    def _apply_chain_inverse(self, x: torch.Tensor, inverses: list) -> torch.Tensor:
        """应用链式逆变换"""
        for inv in inverses:
            x = inv(x)
        return x

    def _mmseg_merge(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """MMSeg合并策略（保持维度正确性）"""
        # 输入校验
        assert all(l.ndim == 3 for l in logits_list), "Logits应为[C, H, W]格式"
        
        # 转换为概率
        probs = [torch.softmax(l, dim=0) for l in logits_list]  # 在dim=0（类别维度）做softmax
        
        # 沿批次维度平均
        stacked = torch.stack(probs, dim=0)
        return stacked.mean(dim=0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整TTA流程
        
        :param image: BGR输入图像 [H, W, 3]
        :return: 分割结果 [H, W]
        """
        # 应用增强
        aug_results = self._apply_augmentations(image)
        
        # 收集并处理所有logits
        logits_list = []
        for aug in aug_results:
            # 获取logits [C, H, W]
            logits = self.segmentor(aug["image"])
            
            # 转换并传输到设备
            logits_tensor = torch.from_numpy(logits).to(self.device)
            
            # 应用逆变换
            logits_tensor = aug["inverse"](logits_tensor)
            
            logits_list.append(logits_tensor)
        
        # 合并结果
        merged = self._mmseg_merge(logits_list)
        
        # 生成最终结果
        result = merged.argmax(dim=0)
            
        return result.cpu().numpy()

# 使用示例
if __name__ == "__main__":
    
    segmentor = Segmentor(model_path="/home/nypyp/code/mmdeploy/models/segnext-l_ade20k_fp16",
                          device_name="cuda",
                          device_id=0)
    
    # 初始化
    tta = TTASegmentor(
        base_segmentor=segmentor,
        device="cuda"
    )
    
    # 处理图像
    img = cv2.imread("input.jpg")
    result = tta.process(img)
    
    # 保存结果
    cv2.imwrite("output.png", result)

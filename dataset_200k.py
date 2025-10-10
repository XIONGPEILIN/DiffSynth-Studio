"""
数据集准备脚本：
1. 从 view_dataset.py 加载数据
2. 将 target 图像的 mask 区域标记为蓝色
3. 使用 FLUX.1-Kontext 消除蓝色区域
4. 保存处理后的数据到指定文件夹
支持多GPU并行处理和多机分布式处理
"""

import os
import json
import torch
import torch.multiprocessing as mp
import random
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import traceback

# 导入 view_dataset.py 中的类
from view_dataset import Subjects200K, make_collate_fn_w_coord


def process_mask_and_image(image_pil, mask_pil, dilation_kernel_size=64):
    """
    用随机颜色填充mask区域，然后扩张填充后的区域，使得物体形状几乎看不出。
    Args:
        image_pil: PIL 图像
        mask_pil: PIL mask 图像
        dilation_kernel_size: 扩张操作的核大小
    Returns:
        处理后的 PIL 图像, 扩张后的 PIL mask
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    if mask_pil.size != image_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)
    
    mask_gray = mask_pil.convert('L')
    
    image_array = np.array(image_pil)
    mask_array = np.array(mask_gray)

    # 定义扩张核
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    # 1. 先用随机颜色填充原始mask区域
    is_foreground = mask_array > 128
    color_filled_image = image_array.copy()
    # 使用更均匀的随机颜色生成（避免过多极值）
    # 方法1: 使用均匀分布
    random_colors = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)
    # 方法2: 使用更温和的高斯噪声（如果想要更自然的变化）
    # noise = np.random.normal(loc=128, scale=50, size=image_array.shape)  
    # random_colors = np.clip(noise, 0, 255).astype(np.uint8)
    color_filled_image[is_foreground] = random_colors[is_foreground]

    # 2. 扩张mask区域（用于标记需要处理的区域）
    dilated_mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    
    # 3. 只在原始mask区域填充随机颜色，扩张区域保持原图
    final_image_array = image_array.copy()
    # 仅在原始mask区域填充随机颜色
    final_image_array[is_foreground] = random_colors[is_foreground]
    # 扩张区域保持原图不变（不填充噪声）
    
    result_image = Image.fromarray(final_image_array)
    dilated_mask_pil = Image.fromarray(dilated_mask_array)
    
    return result_image, dilated_mask_pil


def process_single_sample(args):
    """处理单个样本的函数（用于多进程）"""
    (idx, custom_dataset, output_dirs) = args
    
    tgt_original_dir, tgt_clean_dir, ref_dir = output_dirs
    to_pil_image = transforms.ToPILImage()
    collate_fn = make_collate_fn_w_coord(num_refs=1)
    
    try:
        # 获取样本
        single_item = custom_dataset[idx]
        batch = collate_fn([single_item])
        
        # 提取数据
        sample_id = batch['ids'][0]
        caption = batch['captions'][0]
        
        # 获取原始图像（从 original_dataset）
        example = custom_dataset.original_dataset[idx]
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        # 裁剪 target 和 ref 图像
        padding = custom_dataset.padding
        img_size = custom_dataset.img_size
        
        tgt_img_pil = image.crop(
            (padding, padding, 
             img_size + padding, 
             img_size + padding)
        )
        
        ref_img_pil = image.crop(
            (img_size + padding * 2, padding,
             img_size * 2 + padding * 2, 
             img_size + padding)
        )
        
        # === 获取原始 mask (从数据集中的 tgt_masks) ===
        original_mask_pil = None
        if 'tgt_masks' in batch and len(batch['tgt_masks']) > 0:
            tgt_mask_tensor = batch['tgt_masks'][0]  # [1, H, W]
            original_mask_pil = to_pil_image(tgt_mask_tensor.squeeze(0).float())
        
        # 如果没有原始mask,跳过该样本
        if original_mask_pil is None:
            print(f"\n样本 {idx} (ID: {sample_id}) 没有原始mask,跳过处理")
            return None
        
        # === 保存原始 target 图像 ===
        tgt_img_pil.save(os.path.join(tgt_clean_dir, f"{sample_id}.png"))
        
        # === 处理 target 图像并获取扩张后的 mask ===
        tgt_processed, tgt_dilated_mask = process_mask_and_image(tgt_img_pil, original_mask_pil)
        tgt_processed.save(os.path.join(tgt_original_dir, f"{sample_id}.png"))
        tgt_dilated_mask.save(os.path.join(tgt_original_dir, f"{sample_id}_mask.png"))
        
        # === 保存 ref 图像（不保存 mask） ===
        ref_img_pil.save(os.path.join(ref_dir, f"{sample_id}.png"))
        
        # 记录 metadata
        metadata_entry = {
            "id": sample_id,
            "caption": caption,
            "ref_file": f"{sample_id}.png",
            "ref_mask_file": None,  # 不保存ref mask
            "ref_mask_type": None,  # 不保存ref mask
            "tgt_clean_file": f"{sample_id}.png",  # 原始tgt图片
            "tgt_original_file": f"{sample_id}.png",
            "tgt_original_mask_file": f"{sample_id}_mask.png",
        }
        
        return metadata_entry
        
    except Exception as e:
        print(f"\n处理样本 {idx} 时出错: {e}")
        traceback.print_exc()
        return None


# 用于多进程初始化的全局变量
worker_dataset = None

def init_worker(dataset_args):
    """初始化每个工作进程"""
    global worker_dataset
    dataset, ref_size, tgt_size, grounding_zip, coord_zip = dataset_args
    worker_dataset = Subjects200K(
        original_dataset=dataset,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_zip=grounding_zip,
        coord_zip=coord_zip,
        use_rmbg_comparison=False,
        use_ben_comparison=False
    )

def worker_process(idx_and_dirs):
    """单个样本的处理函数，由进程池调用"""
    return process_single_sample((idx_and_dirs[0], worker_dataset, idx_and_dirs[1]))


def prepare_dataset(
    dataset,
    output_dir="prepared_data_original",
    num_samples=None,
    start_idx=None,
    end_idx=None,
    num_workers=64,
    ref_size=1024,
    tgt_size=1024,
    grounding_zip=None,
    coord_zip=None
):
    """
    准备数据集 - 使用多GPU并行处理，使用数据集原本的mask
    Args:
        dataset: 原始数据集
        output_dir: 输出目录
        num_samples: 处理样本数量（None=全部，优先级低于start_idx/end_idx）
        start_idx: 起始样本索引（用于多机分布式，None=从0开始）
        end_idx: 结束样本索引（用于多机分布式，None=处理到最后）
        num_workers: 使用的CPU线程数
        ref_size: ref 图像大小
        tgt_size: target 图像大小
        grounding_zip: mask zip 文件路径
        coord_zip: coord zip 文件路径
    """
    # 创建输出目录
    tgt_original_dir = os.path.join(output_dir, "tgt_original")
    tgt_clean_dir = os.path.join(output_dir, "tgt_clean")  # 保存原始tgt图片
    ref_dir = os.path.join(output_dir, "ref")
    os.makedirs(tgt_original_dir, exist_ok=True)
    os.makedirs(tgt_clean_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    
    print(f"输出目录:")
    print(f"  - tgt_original: {tgt_original_dir}")
    print(f"  - tgt_clean: {tgt_clean_dir}")
    print(f"  - ref: {ref_dir}")
    
    # 创建临时数据集以获取总样本数
    print("\n创建数据集...")
    temp_dataset = Subjects200K(
        original_dataset=dataset,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_zip=grounding_zip,
        coord_zip=coord_zip,
        use_rmbg_comparison=False,
        use_ben_comparison=False
    )
    print(f"✓ 数据集创建完成，共 {len(temp_dataset)} 个样本")
    
    # 确定处理样本数量和范围
    dataset_size = len(temp_dataset)
    
    # 优先使用 start_idx/end_idx，其次使用 num_samples
    if start_idx is not None or end_idx is not None:
        actual_start = start_idx if start_idx is not None else 0
        actual_end = end_idx if end_idx is not None else dataset_size
        actual_start = max(0, min(actual_start, dataset_size))
        actual_end = max(actual_start, min(actual_end, dataset_size))
        print(f"\n本机处理范围: [{actual_start}, {actual_end}), 共 {actual_end - actual_start} 个样本")
    elif num_samples is not None:
        actual_start = 0
        actual_end = min(num_samples, dataset_size)
        print(f"\n处理前 {actual_end} 个样本")
    else:
        actual_start = 0
        actual_end = dataset_size
        print(f"\n处理全部 {actual_end} 个样本")
    
    indices_to_process = list(range(actual_start, actual_end))
    total_samples = len(indices_to_process)
    print(f"准备使用 {num_workers} 个CPU线程处理 {total_samples} 个样本...")

    # 准备进程池参数
    output_dirs = (tgt_original_dir, tgt_clean_dir, ref_dir)
    worker_args = [(idx, output_dirs) for idx in indices_to_process]
    dataset_args = (dataset, ref_size, tgt_size, grounding_zip, coord_zip)

    # 创建并运行进程池
    metadata = []
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(dataset_args,)) as pool:
        with tqdm(total=total_samples, desc="总体进度") as pbar:
            for result in pool.imap_unordered(worker_process, worker_args):
                if result is not None:
                    metadata.append(result)
                pbar.update(1)

    # 按ID排序metadata以保证顺序
    if metadata:
        # 假设ID可以被转换成数字进行排序
        metadata.sort(key=lambda x: x['id'])

    # 保存 metadata 到 JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据准备完成!")
    print(f"  - 成功处理: {len(metadata)} 个样本")
    print(f"  - Metadata 保存至: {metadata_path}")
    print(f"  - Target (原始未处理) 图像保存至: {tgt_clean_dir}")
    print(f"  - Target (处理后) 图像保存至: {tgt_original_dir}")
    print(f"  - Ref 图像保存至: {ref_dir}")


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    DATA_DIR = "Subjects200K/data"
    GROUNDING_ZIP = "SemAlign-MS-Subjects200K/mask.zip"
    COORD_ZIP = "SemAlign-MS-Subjects200K/coord.zip"
    OUTPUT_DIR = "prepared_data_original"
    
    # ==================== 单机模式 ====================
    NUM_SAMPLES = None  # 处理全部样本
    START_IDX = None    # 从头开始
    END_IDX = None      # 处理到最后
    NUM_WORKERS = 64    # 使用64个CPU线程
    
    # ==================== 多机分布式模式 (4台机器) ====================
    # 数据集总大小: 111767，分配给4台机器，每台约27942个样本
    # 
    # 机器1: 处理 [0, 27942)
    # 机器2: 处理 [27942, 55884)
    # NUM_SAMPLES = None
    # START_IDX = 27942      # 从索引 27942 开始
    # END_IDX = 55884        # 处理到索引 55884
    # NUM_WORKERS = 64       # 使用64个CPU线程
    
    # 机器3: 处理 [55884, 83826)
    # NUM_SAMPLES = None
    # START_IDX = 55884      # 从索引 55884 开始
    # END_IDX = 83826        # 处理到索引 83826
    # NUM_WORKERS = 64       # 使用64个CPU线程
    
    # 机器4: 处理 [83826, 111767]
    # NUM_SAMPLES = None
    # START_IDX = 83826      # 从索引 83826 开始
    # END_IDX = None         # 处理到最后
    # NUM_WORKERS = 64       # 使用64个CPU线程
    
    # 加载原始数据集
    print("加载原始数据集...")
    data_files = {"train": os.listdir(DATA_DIR)}
    dataset = load_dataset(
        "parquet", 
        data_dir=DATA_DIR, 
        data_files=data_files,
    )["train"]
    
    # 过滤数据集
    def filter_func(item):
        if item.get("collection") != "collection_2":
            return False
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    
    print("过滤数据集...")
    dataset_valid = dataset.filter(filter_func, num_proc=16)
    print(f"✓ 过滤后数据集大小: {len(dataset_valid)}")
    
    # 准备数据集 - 使用多GPU并行处理，使用原始mask
    prepare_dataset(
        dataset=dataset_valid,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        start_idx=START_IDX,
        end_idx=END_IDX,
        num_workers=NUM_WORKERS,
        ref_size=1024,
        tgt_size=1024,
        grounding_zip=GROUNDING_ZIP,
        coord_zip=COORD_ZIP
    )

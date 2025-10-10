"""
独立脚本：从原始数据集提取干净的target图片并更新metadata（带图片验证）
功能：
1. 读取现有的metadata.json
2. 从原始数据集中提取对应的target图片（未经处理的原始图片）
3. 验证图片完整性
4. 保存到tgt_clean目录
5. 更新metadata.json，添加edit_image字段
"""

import os
import json
import torch.multiprocessing as mp
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from view_dataset import Subjects200K


# 全局变量，用于多进程访问数据集
_global_dataset = None
_global_original_dataset = None
_global_padding = None
_global_img_size = None


def verify_image(image_path):
    """验证图片是否完整可用"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # 验证图片完整性
        # 再次打开并尝试加载数据（verify后文件会关闭）
        with Image.open(image_path) as img:
            img.load()  # 强制加载图片数据
        return True
    except Exception as e:
        return False


def init_worker_for_extraction(dataset_params):
    """初始化工作进程"""
    global _global_dataset, _global_original_dataset, _global_padding, _global_img_size
    original_dataset, ref_size, tgt_size, grounding_zip, coord_zip = dataset_params
    
    _global_original_dataset = original_dataset
    _global_dataset = Subjects200K(
        original_dataset=original_dataset,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_zip=grounding_zip,
        coord_zip=coord_zip,
        use_rmbg_comparison=False,
        use_ben_comparison=False
    )
    _global_padding = _global_dataset.padding
    _global_img_size = _global_dataset.img_size


def process_single_entry(args):
    """处理单个metadata条目的函数（用于多进程）"""
    entry, tgt_clean_dir, idx = args
    global _global_dataset, _global_original_dataset, _global_padding, _global_img_size
    
    sample_id = entry['id']
    tgt_clean_path = os.path.join(tgt_clean_dir, f"{sample_id}.png")
    
    # 检查是否已经存在且有效
    if os.path.exists(tgt_clean_path):
        # 验证现有文件
        if verify_image(tgt_clean_path):
            # 文件有效，只更新metadata
            if 'edit_image' not in entry:
                entry['edit_image'] = f"tgt_clean/{sample_id}.png"
                return entry, 'updated'
            return entry, 'exists'
        else:
            # 文件损坏，删除并重新提取
            try:
                os.remove(tgt_clean_path)
            except:
                pass
    
    try:
        # 从original_dataset获取原始图像
        example = _global_original_dataset[idx]
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        # 裁剪target图像（原始的，未经处理的）
        tgt_img_pil = image.crop(
            (_global_padding, _global_padding, 
             _global_img_size + _global_padding, 
             _global_img_size + _global_padding)
        )
        
        # 保存原始target图片
        tgt_img_pil.save(tgt_clean_path, format='PNG', optimize=False)
        
        # 验证保存的图片
        if not verify_image(tgt_clean_path):
            # 保存失败，删除损坏的文件
            try:
                os.remove(tgt_clean_path)
            except:
                pass
            return entry, f'error: saved image verification failed'
        
        # 更新metadata
        entry['edit_image'] = f"tgt_clean/{sample_id}.png"
        return entry, 'success'
        
    except Exception as e:
        # 如果出错，尝试删除可能的损坏文件
        if os.path.exists(tgt_clean_path):
            try:
                os.remove(tgt_clean_path)
            except:
                pass
        return entry, f'error: {str(e)}'


def extract_clean_targets(
    original_dataset,
    metadata_path="prepared_data_original/metadata.json",
    output_dir="prepared_data_original",
    ref_size=1024,
    tgt_size=1024,
    grounding_zip="SemAlign-MS-Subjects200K/mask.zip",
    coord_zip="SemAlign-MS-Subjects200K/coord.zip",
    num_workers=32  # 降低并发数，避免资源竞争
):
    """
    从原始数据集提取干净的target图片
    
    Args:
        original_dataset: 原始数据集
        metadata_path: 现有metadata.json的路径
        output_dir: 输出根目录
        ref_size: ref图像尺寸
        tgt_size: target图像尺寸
        grounding_zip: mask zip文件路径
        coord_zip: coord zip文件路径
        num_workers: 使用的CPU线程数
    """
    
    # 创建tgt_clean目录
    tgt_clean_dir = os.path.join(output_dir, "tgt_clean")
    os.makedirs(tgt_clean_dir, exist_ok=True)
    print(f"输出目录: {tgt_clean_dir}")
    
    # 读取现有的metadata
    print(f"\n读取metadata: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✓ 找到 {len(metadata)} 条记录")
    
    # 创建Subjects200K数据集包装器
    print("\n创建数据集包装器...")
    custom_dataset = Subjects200K(
        original_dataset=original_dataset,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_zip=grounding_zip,
        coord_zip=coord_zip,
        use_rmbg_comparison=False,
        use_ben_comparison=False
    )
    print(f"✓ 数据集创建完成，共 {len(custom_dataset)} 个样本")
    
    # 准备数据集参数用于多进程
    print(f"\n准备处理 {len(metadata)} 个样本...")
    dataset_params = (original_dataset, ref_size, tgt_size, grounding_zip, coord_zip)
    
    # 准备处理参数
    process_args = [
        (entry, tgt_clean_dir, idx)
        for idx, entry in enumerate(metadata)
    ]
    
    # 使用多进程处理
    print(f"\n开始提取target图片 (使用 {num_workers} 个线程)...")
    updated_metadata = []
    success_count = 0
    exists_count = 0
    updated_count = 0
    not_found_count = 0
    error_count = 0
    error_details = []
    
    with mp.Pool(processes=num_workers, initializer=init_worker_for_extraction, initargs=(dataset_params,)) as pool:
        with tqdm(total=len(metadata), desc="总体进度") as pbar:
            for result_entry, status in pool.imap_unordered(process_single_entry, process_args):
                updated_metadata.append(result_entry)
                
                if status == 'success':
                    success_count += 1
                elif status == 'exists':
                    exists_count += 1
                elif status == 'updated':
                    updated_count += 1
                elif status == 'not_found':
                    not_found_count += 1
                elif status.startswith('error'):
                    error_count += 1
                    error_details.append({
                        'id': result_entry['id'],
                        'error': status
                    })
                
                pbar.update(1)
    
    # 按ID排序metadata以保证顺序
    updated_metadata.sort(key=lambda x: x['id'])
    
    # 保存更新后的metadata
    print(f"\n保存更新后的metadata...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
    
    # 如果有错误，保存错误详情
    if error_details:
        error_log_path = os.path.join(output_dir, "extraction_errors.json")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(error_details, f, indent=2, ensure_ascii=False)
        print(f"  - 错误详情已保存至: {error_log_path}")
    
    print(f"\n✅ 完成!")
    print(f"  - 成功提取并保存: {success_count} 个样本")
    print(f"  - 已存在(仅更新metadata): {updated_count} 个样本")
    print(f"  - 文件已存在(跳过): {exists_count} 个样本")
    print(f"  - ID未找到: {not_found_count} 个样本")
    print(f"  - 处理错误: {error_count} 个样本")
    print(f"  - Target原始图片保存至: {tgt_clean_dir}")
    print(f"  - Metadata已更新: {metadata_path}")
    
    if error_count > 0:
        print(f"\n⚠️  警告: 有 {error_count} 个样本提取失败")
        print(f"  建议运行文件检查脚本清理损坏的条目")


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    DATA_DIR = "Subjects200K/data"
    GROUNDING_ZIP = "SemAlign-MS-Subjects200K/mask.zip"
    COORD_ZIP = "SemAlign-MS-Subjects200K/coord.zip"
    METADATA_PATH = "prepared_data_original/metadata.json"
    OUTPUT_DIR = "prepared_data_original"
    NUM_WORKERS = 32  # 降低为32，更稳定
    
    # 加载原始数据集
    print("加载原始数据集...")
    data_files = {"train": os.listdir(DATA_DIR)}
    dataset = load_dataset(
        "parquet", 
        data_dir=DATA_DIR, 
        data_files=data_files,
    )["train"]
    
    # 过滤数据集（需要与dataset_200k.py保持一致）
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
    
    # 提取干净的target图片
    extract_clean_targets(
        original_dataset=dataset_valid,
        metadata_path=METADATA_PATH,
        output_dir=OUTPUT_DIR,
        ref_size=1024,
        tgt_size=1024,
        grounding_zip=GROUNDING_ZIP,
        coord_zip=COORD_ZIP,
        num_workers=NUM_WORKERS
    )
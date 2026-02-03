"""
DPDD 数据集预处理/验证脚本
================================
验证 train_c, val_c, test_c 三个文件夹的数据完整性。

由于原始图像尺寸 (1680x1120) 已接近目标尺寸，不再进行 resize 和重存操作，
直接使用原始数据以保留图像质量。

DPDD Canon Set 数据量:
- train_c: 350 对图像 (训练集)
- val_c:   74 对图像 (验证集)  
- test_c:  76 对图像 (测试集)

Usage:
    python data/preprocess_dpdd.py
"""

import os
from PIL import Image
from tqdm import tqdm


def validate_dpdd_dataset():
    """
    验证 DPDD 数据集的完整性，不进行任何图像处理。
    直接使用原始数据文件夹结构。
    """
    # 数据集根路径
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dd_dp_dataset_png")
    
    # 三个数据集分区
    splits = {
        'train': 'train_c',  # 训练集
        'val': 'val_c',      # 验证集
        'test': 'test_c'     # 测试集
    }
    
    expected_counts = {
        'train': 350,
        'val': 74,
        'test': 76
    }
    
    valid_exts = ('.png', '.tif', '.jpg', '.jpeg', '.tiff')
    
    print("="*60)
    print("DPDD Dataset Validation")
    print("="*60)
    print(f"Dataset Root: {DATASET_ROOT}")
    print()
    
    all_valid = True
    dataset_info = {}
    
    for split_name, folder_name in splits.items():
        split_dir = os.path.join(DATASET_ROOT, folder_name)
        source_dir = os.path.join(split_dir, "source")  # 模糊图像
        target_dir = os.path.join(split_dir, "target")  # 清晰图像
        
        print(f"\n[{split_name.upper()}] Checking {folder_name}...")
        
        # 检查目录是否存在
        if not os.path.exists(split_dir):
            print(f"  ✗ ERROR: Directory not found: {split_dir}")
            all_valid = False
            continue
            
        if not os.path.exists(source_dir):
            print(f"  ✗ ERROR: Source directory not found: {source_dir}")
            all_valid = False
            continue
            
        if not os.path.exists(target_dir):
            print(f"  ✗ ERROR: Target directory not found: {target_dir}")
            all_valid = False
            continue
        
        # 获取文件列表
        source_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)])
        target_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(valid_exts)])
        
        n_source = len(source_files)
        n_target = len(target_files)
        expected = expected_counts.get(split_name, 0)
        
        print(f"  Source (blur) images:  {n_source}")
        print(f"  Target (sharp) images: {n_target}")
        print(f"  Expected count:        {expected}")
        
        # 验证数量匹配
        if n_source != n_target:
            print(f"  ✗ ERROR: Mismatch between source and target counts!")
            all_valid = False
        elif n_source != expected:
            print(f"  ⚠ WARNING: Count mismatch with expected ({expected})")
        else:
            print(f"  ✓ Count OK")
        
        # 检查图像尺寸（采样几张）
        sample_indices = [0, n_source // 2, n_source - 1] if n_source >= 3 else range(n_source)
        sizes_ok = True
        sample_size = None
        
        for idx in sample_indices:
            if idx < len(source_files):
                src_path = os.path.join(source_dir, source_files[idx])
                try:
                    with Image.open(src_path) as img:
                        w, h = img.size
                        if sample_size is None:
                            sample_size = (w, h)
                        elif (w, h) != sample_size:
                            sizes_ok = False
                            print(f"  ⚠ WARNING: Inconsistent sizes detected")
                except Exception as e:
                    print(f"  ✗ ERROR reading {src_path}: {e}")
                    all_valid = False
        
        if sample_size:
            print(f"  Image size: {sample_size[0]}x{sample_size[1]}")
            if sizes_ok:
                print(f"  ✓ Image sizes consistent")
        
        # 存储数据集信息
        dataset_info[split_name] = {
            'folder': folder_name,
            'source_dir': source_dir,
            'target_dir': target_dir,
            'count': n_source,
            'size': sample_size,
            'source_files': source_files,
            'target_files': target_files
        }
    
    # 总结
    print("\n" + "="*60)
    if all_valid:
        print("✓ Dataset validation PASSED!")
        print("\nDataset is ready for direct loading (no preprocessing needed).")
        print("\nSummary:")
        for split_name, info in dataset_info.items():
            print(f"  - {split_name}: {info['count']} pairs, size={info['size']}")
    else:
        print("✗ Dataset validation FAILED!")
        print("Please check the errors above and fix the dataset structure.")
    print("="*60)
    
    return all_valid, dataset_info


def get_image_pairs(split='train'):
    """
    获取指定分区的图像对路径列表。
    
    Args:
        split: 'train', 'val', 或 'test'
    
    Returns:
        list of dict: [{'source': path_to_blur, 'target': path_to_sharp}, ...]
    """
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dd_dp_dataset_png")
    
    splits_map = {
        'train': 'train_c',
        'val': 'val_c',
        'test': 'test_c'
    }
    
    if split not in splits_map:
        raise ValueError(f"Invalid split '{split}'. Must be one of {list(splits_map.keys())}")
    
    folder_name = splits_map[split]
    split_dir = os.path.join(DATASET_ROOT, folder_name)
    source_dir = os.path.join(split_dir, "source")
    target_dir = os.path.join(split_dir, "target")
    
    valid_exts = ('.png', '.tif', '.jpg', '.jpeg', '.tiff')
    source_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)])
    target_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(valid_exts)])
    
    if len(source_files) != len(target_files):
        raise ValueError(f"Mismatch in {split}: {len(source_files)} source vs {len(target_files)} target")
    
    pairs = []
    for src_f, tgt_f in zip(source_files, target_files):
        pairs.append({
            'source': os.path.join(source_dir, src_f),
            'target': os.path.join(target_dir, tgt_f),
            'name': os.path.splitext(src_f)[0]
        })
    
    return pairs


if __name__ == "__main__":
    validate_dpdd_dataset()

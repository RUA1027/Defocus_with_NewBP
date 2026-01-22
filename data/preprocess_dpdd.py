import os
from PIL import Image
from tqdm import tqdm

def preprocess_dpdd():
    # 1. 路径配置 (硬编码)
    # 修正路径：根据工作区实际结构，指向 'test_c' (Canon Test Set)
    SOURCE_ROOT = r"D:\Defocus_with_NewBP\data\dd_dp_dataset_png\test_c" 
    OUTPUT_ROOT = r"D:\Defocus_with_NewBP\data\dpdd_1024"

    # 定义输入子目录
    input_blur_dir = os.path.join(SOURCE_ROOT, "source")
    input_sharp_dir = os.path.join(SOURCE_ROOT, "target")

    # 检查输入源是否存在
    if not os.path.exists(input_blur_dir) or not os.path.exists(input_sharp_dir):
        print(f"Error: Source directories not found in {SOURCE_ROOT}")
        print("Expected 'source' and 'target' subdirectories.")
        print(f"Current Source Root: {SOURCE_ROOT}")
        return

    # 2. 获取并排序文件列表
    print("Scanning files...")
    # 过滤图片扩展名
    valid_exts = ('.png', '.tif', '.jpg', '.jpeg', '.tiff')
    
    # 分别获取 source 和 target 的文件列表
    blur_files = sorted([f for f in os.listdir(input_blur_dir) if f.lower().endswith(valid_exts)])
    sharp_files = sorted([f for f in os.listdir(input_sharp_dir) if f.lower().endswith(valid_exts)])

    total_blur = len(blur_files)
    total_sharp = len(sharp_files)
    print(f"Found {total_blur} source images and {total_sharp} target images.")

    if total_blur != total_sharp:
        print("Error: Mismatch in number of source and target images!")
        return
    
    # 配对文件 (假设排序后一一对应)
    # DPDD 数据集中，source 和 target 文件名通常不同 (连续拍摄)，需要按顺序配对
    files_map = []
    for i in range(total_blur):
        blur_f = blur_files[i]
        sharp_f = sharp_files[i]
        
        # 简单校验一下，文件名应该如果不一致，但必须是同一个场景
        # 这里只做打印提示，不强制报错，信任排序结果
        if i < 3: 
            print(f"  Pair {i}: Blur('{blur_f}') -> Sharp('{sharp_f}')")

        files_map.append({
            "name": blur_f, # 使用模糊图名称作为 ID
            "blur_path": os.path.join(input_blur_dir, blur_f),
            "sharp_path": os.path.join(input_sharp_dir, sharp_f)
        })

    total_files = len(files_map)
    
    # 3. 数据集划分
    # Train: 0-59 (60张), Val: 60-63 (4张), Test: 64-75 (12张)
    train_objs = files_map[:60]
    val_objs = files_map[60:64]
    test_objs = files_map[64:76]
    
    print(f"Split Summary:")
    print(f"  Train: {len(train_objs)} (Indices 0-59)")
    print(f"  Val:   {len(val_objs)}   (Indices 60-63)")
    print(f"  Test:  {len(test_objs)}  (Indices 64-75)")

    splits = [
        ("train", train_objs),
        ("val", val_objs),
        ("test", test_objs)
    ]

    # 4. 图像处理与保存
    target_size = (1024, 1024)
    # 兼容不同 PIL 版本
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS

    for split_name, split_obj_list in splits:
        print(f"\nProcessing {split_name} set...")
        
        # 创建输出目录结构
        out_blur_dir = os.path.join(OUTPUT_ROOT, split_name, "blur")
        out_sharp_dir = os.path.join(OUTPUT_ROOT, split_name, "sharp")
        
        os.makedirs(out_blur_dir, exist_ok=True)
        os.makedirs(out_sharp_dir, exist_ok=True)

        for item in tqdm(split_obj_list, desc=f"Converting {split_name}"):
            src_blur_path = item["blur_path"]
            src_sharp_path = item["sharp_path"]
            filename = item["name"]

            # 读取图片
            # 使用 .convert('RGB') 确保兼容性
            with Image.open(src_blur_path) as img_blur:
                # 转换 RGB 防止 RGBA 问题
                if img_blur.mode != 'RGB':
                    img_blur = img_blur.convert('RGB')
                img_blur_resized = img_blur.resize(target_size, resample=resample_method)
                
                name_no_ext = os.path.splitext(filename)[0]
                save_path_blur = os.path.join(out_blur_dir, name_no_ext + ".png")
                img_blur_resized.save(save_path_blur, "PNG")

            with Image.open(src_sharp_path) as img_sharp:
                if img_sharp.mode != 'RGB':
                    img_sharp = img_sharp.convert('RGB')
                img_sharp_resized = img_sharp.resize(target_size, resample=resample_method)
                
                name_no_ext = os.path.splitext(filename)[0] # 重复计算一下避免作用域问题
                save_path_sharp = os.path.join(out_sharp_dir, name_no_ext + ".png")
                img_sharp_resized.save(save_path_sharp, "PNG")

    print("\n" + "="*50)
    print(f"Done! Dataset generated at: {OUTPUT_ROOT}")
    print("="*50)

if __name__ == "__main__":
    preprocess_dpdd()

import numpy as np
from PIL import Image
import cv2

def process_mask_and_image_no_noise_expansion(image_pil, mask_pil, dilation_kernel_size=50):
    """
    修改版：只在原始mask区域填充噪声，扩张区域保持原图
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
    # 使用均匀分布生成随机颜色
    random_colors = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)
    
    # 2. 扩张mask区域（用于标记需要处理的区域）
    dilated_mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    
    # 3. 只在原始mask区域填充随机颜色，扩张区域保持原图
    final_image_array = image_array.copy()
    # 仅在原始mask区域填充随机颜色
    final_image_array[is_foreground] = random_colors[is_foreground]
    # 扩张区域保持原图不变（不填充噪声）
    
    # 统计信息
    print(f"原始mask区域像素数: {np.sum(is_foreground)}")
    print(f"扩张后mask区域像素数: {np.sum(dilated_mask_array > 128)}")
    print(f"扩张增加的像素数: {np.sum(dilated_mask_array > 128) - np.sum(is_foreground)}")
    
    if np.sum(is_foreground) > 0:
        # 检查原始mask区域是否成功填充噪声
        original_colors = final_image_array[is_foreground]
        print(f"原始mask区域填充后的颜色统计:")
        print(f"  - 均值: {original_colors.mean():.2f}")
        print(f"  - 标准差: {original_colors.std():.2f}")
        print(f"  - 最小值: {original_colors.min()}")
        print(f"  - 最大值: {original_colors.max()}")
    
    # 检查扩张区域是否保持原图
    dilated_mask_bool = dilated_mask_array > 128
    expanded_area = dilated_mask_bool & ~is_foreground
    if np.sum(expanded_area) > 0:
        # 检查扩张区域的颜色（应该是原图颜色）
        expanded_colors = final_image_array[expanded_area]
        original_expanded_colors = image_array[expanded_area]
        
        # 检查是否完全一致
        is_same = np.array_equal(expanded_colors, original_expanded_colors)
        print(f"扩张区域是否保持原图: {is_same}")
        if not is_same:
            print(f"  - 扩张区域均值: {expanded_colors.mean():.2f}")
            print(f"  - 原图扩张区域均值: {original_expanded_colors.mean():.2f}")
    
    result_image = Image.fromarray(final_image_array)
    dilated_mask_pil = Image.fromarray(dilated_mask_array)
    
    return result_image, dilated_mask_pil

# 创建测试图像和mask
print("测试修改后的版本（扩张区域不填充噪声）...")

# 创建一个更有特征的测试图像
test_image_array = np.zeros((256, 256, 3), dtype=np.uint8)
# 背景渐变（从红到蓝）
for i in range(256):
    test_image_array[:, i, 0] = 255 - i  # 红色通道递减
    test_image_array[:, i, 2] = i       # 蓝色通道递增
test_image = Image.fromarray(test_image_array)

# 创建圆形mask
test_mask = Image.new('L', (256, 256), color=0)
mask_array = np.array(test_mask)
center_x, center_y = 128, 128
radius = 50

# 创建圆形mask
y, x = np.ogrid[:256, :256]
mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
mask_array[mask_circle] = 255
test_mask = Image.fromarray(mask_array)

# 测试修改后的处理函数
result_img, result_mask = process_mask_and_image_no_noise_expansion(test_image, test_mask, dilation_kernel_size=30)

# 保存结果
test_image.save("/home2/y2024/s2430069/DiffSynth-Studio/test_gradient_original.png")
test_mask.save("/home2/y2024/s2430069/DiffSynth-Studio/test_gradient_mask.png")
result_img.save("/home2/y2024/s2430069/DiffSynth-Studio/test_gradient_result.png")
result_mask.save("/home2/y2024/s2430069/DiffSynth-Studio/test_gradient_result_mask.png")

print("\n测试完成！查看以下文件:")
print("  - test_gradient_original.png: 原始渐变图像")
print("  - test_gradient_mask.png: 原始mask")
print("  - test_gradient_result.png: 处理后图像（只有圆形区域填充噪声）")
print("  - test_gradient_result_mask.png: 扩张后mask")
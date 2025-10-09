import os
import random
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import hashlib
from io import BytesIO
import traceback
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import zipfile


def get_image_hash(image: Image.Image) -> str:
    """Convert a PIL image to a SHA256 hash string."""
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
    return hashlib.sha256(img_bytes).hexdigest()[:8]  # 取前8位更短但足够唯一


def get_mask_bbox_area(mask):
    """
    计算mask的bounding box面积
    Args:
        mask: PIL Image 或 torch.Tensor，二值化mask
    Returns:
        int: bounding box的面积 (width * height)
    """
    if isinstance(mask, Image.Image):
        # PIL Image转为numpy
        mask_array = np.array(mask)
    elif isinstance(mask, torch.Tensor):
        # torch.Tensor转为numpy
        if mask.dim() == 3:  # [C, H, W]
            mask_array = mask.squeeze(0).numpy()
        else:  # [H, W]
            mask_array = mask.numpy()
    else:
        mask_array = np.array(mask)
    
    # 二值化处理
    if mask_array.dtype != bool:
        mask_array = mask_array > 0.5
    
    # 找到非零像素的坐标
    nonzero_coords = np.where(mask_array)
    
    if len(nonzero_coords[0]) == 0:
        # 如果没有非零像素，返回0
        return 0
    
    # 计算bounding box
    min_y, max_y = nonzero_coords[0].min(), nonzero_coords[0].max()
    min_x, max_x = nonzero_coords[1].min(), nonzero_coords[1].max()
    
    # 计算面积
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    area = width * height
    
    return area


def select_smaller_bbox_mask(original_mask, predicted_mask):
    """
    比较两个mask的bbox大小，选择bbox更小的那个
    Args:
        original_mask: 原始mask (PIL Image 或 torch.Tensor)
        predicted_mask: 预测的mask (PIL Image 或 torch.Tensor)
    Returns:
        选择的mask, bbox面积, 选择的类型 ("original" 或 "predicted")
    """
    original_area = get_mask_bbox_area(original_mask)
    predicted_area = get_mask_bbox_area(predicted_mask)
    
    if original_area <= predicted_area:
        return original_mask, original_area, "original"
    else:
        return predicted_mask, predicted_area, "predicted"


def generate_rmbg_mask(pil_image, rmbg_model, rmbg_transform, device):
    """
    使用RMBG模型生成mask
    Args:
        pil_image: PIL图像
        rmbg_model: RMBG模型
        rmbg_transform: RMBG预处理转换
        device: 设备
    Returns:
        PIL mask图像
    """
    # 确保图像是RGB模式(不是RGBA)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    original_size = pil_image.size
    input_tensor = rmbg_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = rmbg_model(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size, Image.Resampling.LANCZOS)
    return mask


def generate_ben_mask(pil_image, ben_model):
    """
    使用BEN模型生成mask
    Args:
        pil_image: PIL图像
        ben_model: BEN模型
    Returns:
        PIL mask图像（灰度图，白色=前景，黑色=背景）
    """
    try:
        # BEN模型接受PIL图像列表，返回前景mask列表（带透明通道的RGBA图像）
        foregrounds = ben_model.inference([pil_image])
        foreground_rgba = foregrounds[0]
        
        # 将RGBA图像的alpha通道转换为灰度mask
        if foreground_rgba.mode == 'RGBA':
            # 提取alpha通道作为mask
            alpha_channel = foreground_rgba.split()[3]  # A通道
            # alpha通道: 255=不透明(前景), 0=透明(背景)
            # 转换为mask: 255=前景(白色), 0=背景(黑色)
            mask = alpha_channel.convert('L')
            return mask
        elif foreground_rgba.mode == 'L':
            # 如果已经是灰度图，直接返回
            return foreground_rgba
        else:
            # 其他模式，转换为灰度
            return foreground_rgba.convert('L')
    except Exception as e:
        print(f"BEN mask generation failed: {e}")
        # 如果失败，返回None
        return None


def select_smallest_bbox_mask(*masks_with_names):
    """
    比较多个mask的bbox大小，选择bbox最小的那个
    Args:
        *masks_with_names: 元组列表，每个元组包含 (mask, name)
    Returns:
        选择的mask, bbox面积, 选择的类型名称, 所有面积字典
    """
    if not masks_with_names:
        return None, 0, "none", {}
    
    min_area = float('inf')
    selected_mask = None
    selected_name = None
    
    areas = {}
    
    for mask, name in masks_with_names:
        if mask is not None:
            area = get_mask_bbox_area(mask)
            areas[name] = area
            if area < min_area:
                min_area = area
                selected_mask = mask
                selected_name = name
    
    return selected_mask, min_area, selected_name, areas


def build_ref2tgt_mapping(
        dift_coord, geoaware_coord, mask, 
        tgt_patch_size=64, ref_patch_size=32
    ):
    """
    Args:
        dift_coord: Tensor [N, 2], tgt_patch -> ref_patch (y, x) on ref grid
        geoaware_coord: Tensor [N, 2], tgt_patch -> ref_patch (y, x) on ref grid
        mask: Tensor [1, H, W] (binary), tgt mask (already resized to tgt_size)
        tgt_patch_size: int, tgt patch grid resolution (default 64 for 1024/16)
        ref_patch_size: int, ref patch grid resolution (if None, assume same as tgt)

    Returns:
        ref2tgt: dict {ref_patch_id: [tgt_patch_id, ...]}
    """
    N = dift_coord.shape[0]   # tgt patch count
    assert dift_coord.shape == geoaware_coord.shape

    if ref_patch_size is None:
        ref_patch_size = tgt_patch_size  # 默认和 tgt 相同

    # --- 1. 找出交集位置 ---
    valid_mask = torch.all(dift_coord == geoaware_coord, dim=1)  # [N] bool

    # --- 2. resize mask 到 tgt patch 空间 ---
    mask_patch = F.interpolate(mask.unsqueeze(0), size=(tgt_patch_size, tgt_patch_size),
                            mode="nearest")[0,0]   # [tgt_patch_size, tgt_patch_size]
    mask_flat = mask_patch.view(-1) > 0   # [N] bool

    # --- 3. tgt patch 中有效的索引 = 两个条件同时成立
    keep_mask = valid_mask & mask_flat.to(device=dift_coord.device)
    keep_idx = torch.arange(N, device=dift_coord.device)[keep_mask]  # [K], 对应 tgt 中对应物体有效 idx 区域

    # 4. ref 坐标 (64-grid → target_ref_patch_size-grid)
    ref_coords64 = dift_coord[keep_idx]  # [K,2]
    ref_coords_new = (ref_coords64 * ref_patch_size) // 64
    ref_patch_idx = ref_coords_new[:,0] * ref_patch_size + ref_coords_new[:,1]

    # 5. 构建反映射
    ref2tgt = {}
    for r, t in zip(ref_patch_idx.tolist(), keep_idx.tolist()):
        if r not in ref2tgt:   # 只保留第一个
            ref2tgt[r] = t

    return ref2tgt


class Subjects200K(Dataset):
    def __init__(
        self, 
        original_dataset, 
        mode, 
        ref_size, 
        tgt_size, 
        grounding_dir=None,
        grounding_zip=None,
        coord_folder: str=None,
        coord_zip: str=None,
        padding=8, 
        img_size=512, 
        t_drop_rate=0.05, 
        i_drop_rate=0.05, 
        ti_drop_rate=0.05,
        add_postfix=False,
        use_rmbg_comparison=False,  # 是否使用RMBG mask比较
        use_ben_comparison=False    # 新增参数：是否使用BEN mask比较
    ):
        self.original_dataset = original_dataset
        self.grounding_dir = grounding_dir
        self.grounding_zip = grounding_zip
        self.coord_folder = coord_folder
        self.coord_zip = coord_zip
        
        # 如果提供了 coord_zip，则打开它
        self.coord_zip_file = None
        if self.coord_zip and os.path.exists(self.coord_zip):
            self.coord_zip_file = zipfile.ZipFile(self.coord_zip, 'r')
        
        # 如果提供了 grounding_zip，则打开它
        self.grounding_zip_file = None
        if self.grounding_zip and os.path.exists(self.grounding_zip):
            self.grounding_zip_file = zipfile.ZipFile(self.grounding_zip, 'r')
            
        self.mode = mode
        self.padding = padding
        self.ref_size = ref_size # resize 大小
        self.tgt_size = tgt_size # resize 大小
        self.img_size = img_size # 原图大小
        
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.add_postfix = add_postfix
        self.use_rmbg_comparison = use_rmbg_comparison
        self.use_ben_comparison = use_ben_comparison
        
        # 如果启用RMBG比较，初始化RMBG模型
        self.rmbg_model = None
        self.rmbg_transform = None
        self.device = None
        if self.use_rmbg_comparison:
            try:
                from transformers import AutoModelForImageSegmentation
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                    'briaai/RMBG-2.0', trust_remote_code=True
                ).eval().to(self.device)
                
                self.rmbg_transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print(f"RMBG-2.0 模型已加载到 {self.device}")
            except ImportError:
                print("警告: 未找到 `transformers` 库，禁用 RMBG mask 比较功能")
                self.use_rmbg_comparison = False
            except Exception as e:
                print(f"警告: 加载 RMBG 模型失败: {e}")
                self.use_rmbg_comparison = False
        
        # 如果启用BEN比较，初始化BEN模型
        self.ben_model = None
        if self.use_ben_comparison:
            try:
                from ben2 import BEN_Base
                if self.device is None:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.ben_model = BEN_Base.from_pretrained("PramaLLC/BEN2")
                self.ben_model.to(self.device).eval()
                print(f"BEN 模型已加载到 {self.device}")
            except ImportError:
                print("警告: 未找到 `ben2` 库，禁用 BEN mask 比较功能")
                self.use_ben_comparison = False
            except Exception as e:
                print(f"警告: 加载 BEN 模型失败: {e}")
                self.use_ben_comparison = False
        self.transform_ref = transforms.Compose([   
            transforms.Resize(self.ref_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.ToTensor(), # [0, 255] --> [0, 1]
            transforms.Normalize([0.5], [0.5]), # [0, 1] --> [-1, 1]
        ])

        self.transform_tgt = transforms.Compose([   
            transforms.Resize(self.tgt_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.ToTensor(), # [0, 255] --> [0, 1]
            transforms.Normalize([0.5], [0.5]), # [0, 1] --> [-1, 1]
        ])

        self.transform_mask_tgt = transforms.Compose([
            transforms.Resize(self.tgt_size, interpolation=transforms.InterpolationMode.NEAREST),  # 保留 mask 边界
            transforms.ToTensor(),  # 转为 [0,1] 的 float tensor, shape: [1, H, W]
            transforms.Lambda(lambda x: (x > 0.5).to(torch.uint8))  # 二值化，0 或 1
        ])
        
        self.transform_mask_ref = transforms.Compose([
            transforms.Resize(self.ref_size, interpolation=transforms.InterpolationMode.NEAREST),  # 保留 mask 边界
            transforms.ToTensor(),  # 转为 [0,1] 的 float tensor, shape: [1, H, W]
            transforms.Lambda(lambda x: (x > 0.5).to(torch.uint8))  # 二值化，0 或 1
        ])

    def _compare_and_select_mask(self, img_pil, uid, mask_type_prefix="", mask_transform=None):
        """
        比较原始mask、RMBG mask和BEN mask，选择bbox最小的
        Args:
            img_pil: PIL图像
            uid: 图像唯一标识符
            mask_type_prefix: mask类型前缀（如"tgt"或"ref"）
            mask_transform: mask转换函数（用于resize到对应大小）
        Returns:
            (选择的mask tensor, 选择的类型名称)，如果没有原始mask则返回(None, None)
        """
        if mask_transform is None:
            mask_transform = self.transform_mask_tgt  # 默认使用tgt的transform
        # 尝试加载原始mask（按照原版代码的逻辑）
        original_mask_pil = None
        
        if self.grounding_zip_file:
            # 从 zip 文件中读取
            mask_path_in_zip = f"mask/{uid}.png"
            try:
                with self.grounding_zip_file.open(mask_path_in_zip) as f:
                    original_mask_pil = Image.open(BytesIO(f.read())).convert("L")
            except KeyError:
                pass  # 文件不存在
        elif self.grounding_dir:
            # 从文件夹读取（原版代码方式）
            mask_path = os.path.join(self.grounding_dir, f"{uid}.png")
            if os.path.exists(mask_path):
                original_mask_pil = Image.open(mask_path).convert("L")
        
        if original_mask_pil is None:
            return None, None
            
        # 如果没有启用任何比较，直接使用原始mask
        if not (self.use_rmbg_comparison or self.use_ben_comparison):
            return mask_transform(original_mask_pil), "original"
        
        # 如果启用mask比较，生成其他模型的mask并比较
        if (self.use_rmbg_comparison and self.rmbg_model is not None) or \
           (self.use_ben_comparison and self.ben_model is not None):
            try:
                masks_to_compare = [(original_mask_pil, "original")]
                
                # 生成RMBG mask
                if self.use_rmbg_comparison and self.rmbg_model is not None:
                    rmbg_mask_pil = generate_rmbg_mask(
                        img_pil, self.rmbg_model, self.rmbg_transform, self.device
                    )
                    if rmbg_mask_pil is not None:
                        masks_to_compare.append((rmbg_mask_pil, "rmbg"))
                
                # 生成BEN mask
                if self.use_ben_comparison and self.ben_model is not None:
                    ben_mask_pil = generate_ben_mask(img_pil, self.ben_model)
                    if ben_mask_pil is not None:
                        masks_to_compare.append((ben_mask_pil, "ben"))
                
                # 比较所有mask，选择bbox最小的
                if len(masks_to_compare) > 1:
                    selected_mask_pil, bbox_area, mask_type, all_areas = select_smallest_bbox_mask(*masks_to_compare)
                    
                    if selected_mask_pil is not None:
                        # 打印比较结果
                        areas_str = ", ".join([f"{name}={area}" for name, area in all_areas.items()])
                        print(f"Mask comparison for {mask_type_prefix}_{uid}: {areas_str}, selected={mask_type}")
                        
                        # 可选：保存选择的mask用于调试
                        # selected_mask_pil.save(f"selected_mask_{mask_type_prefix}_{uid}_{mask_type}.png")
                        
                        return mask_transform(selected_mask_pil), mask_type
                
            except Exception as e:
                print(f"Mask comparison failed for {mask_type_prefix}_{uid}: {e}")
        
        # 如果比较失败或没有其他mask，使用原始mask
        return mask_transform(original_mask_pil), "original"

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        try:
            example = self.original_dataset[idx]
            image = example['image']
            description = example['description']
            item = description['item']
            tgt_caption = description['description_0']
            ref_caption = description['description_1']
            
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")

            tgt_img_pil = image.crop(
                (self.padding, self.padding, self.img_size + self.padding, self.img_size + self.padding)
            )
            ref_img_pil = image.crop(
                (
                    self.img_size + self.padding * 2,
                    self.padding,
                    self.img_size * 2 + self.padding * 2,
                    self.img_size + self.padding,
                )
            )
            # ref_img = self.transform(ref_img)
            ref_img = self.transform_ref(ref_img_pil)
            tgt_img = self.transform_tgt(tgt_img_pil)

            # 生成uid（按照原版代码的逻辑）
            try:
                img_hash = get_image_hash(tgt_img_pil)
            except Exception as e:
                raise ValueError(f"Failed to hash image: {e}")
            
            key = f"{item}_{tgt_caption}_{img_hash}"
            uid = hashlib.sha256(key.encode("utf-8")).hexdigest()
            
            # 为目标图像使用三种mask比较方法
            mask_tgt_tensor, mask_tgt_type = self._compare_and_select_mask(tgt_img_pil, uid, "tgt", self.transform_mask_tgt)
            if mask_tgt_tensor is not None:
                mask_tgt = mask_tgt_tensor
            else:
                mask_tgt = torch.ones((1, self.tgt_size, self.tgt_size), dtype=torch.uint8)
                mask_tgt_type = "default"
            
            # 为参考图像尝试使用mask（ref图像使用uid+"_ref0"后缀）
            mask_ref_tensor, mask_ref_type = self._compare_and_select_mask(ref_img_pil, f"{uid}_ref0", "ref", self.transform_mask_ref)
            
            if mask_ref_tensor is not None:
                mask_ref = mask_ref_tensor
            else:
                # 如果找不到ref的原始mask，生成ref图像的mask（不使用原始mask）
                mask_ref_type = "default"
                if (self.use_rmbg_comparison and self.rmbg_model is not None) or \
                   (self.use_ben_comparison and self.ben_model is not None):
                    try:
                        masks_to_compare = []
                        
                        # 生成RMBG mask
                        if self.use_rmbg_comparison and self.rmbg_model is not None:
                            rmbg_mask_pil = generate_rmbg_mask(
                                ref_img_pil, self.rmbg_model, self.rmbg_transform, self.device
                            )
                            if rmbg_mask_pil is not None:
                                masks_to_compare.append((rmbg_mask_pil, "rmbg"))
                        
                        # 生成BEN mask
                        if self.use_ben_comparison and self.ben_model is not None:
                            ben_mask_pil = generate_ben_mask(ref_img_pil, self.ben_model)
                            if ben_mask_pil is not None:
                                masks_to_compare.append((ben_mask_pil, "ben"))
                        
                        if masks_to_compare:
                            selected_mask_pil, bbox_area, mask_ref_type, all_areas = select_smallest_bbox_mask(*masks_to_compare)
                            if selected_mask_pil is not None:
                                areas_str = ", ".join([f"{name}={area}" for name, area in all_areas.items()])
                                print(f"Mask comparison for ref_{uid}_ref0: {areas_str}, selected={mask_ref_type}")
                                mask_ref = self.transform_mask_ref(selected_mask_pil)
                            else:
                                mask_ref = torch.ones((1, self.ref_size, self.ref_size), dtype=torch.uint8)
                        else:
                            mask_ref = torch.ones((1, self.ref_size, self.ref_size), dtype=torch.uint8)
                    except Exception as e:
                        print(f"Failed to generate ref mask: {e}")
                        mask_ref = torch.ones((1, self.ref_size, self.ref_size), dtype=torch.uint8)
                else:
                    mask_ref = torch.ones((1, self.ref_size, self.ref_size), dtype=torch.uint8)

            coords = []
            
            # 尝试从 zip 或文件夹加载坐标
            dift_coord = None
            geoaware_coord = None
            
            if self.coord_zip_file:
                # 从 zip 文件中读取
                dift_path_in_zip = f"coord/dift/{uid}_ref0.pt"
                geoaware_path_in_zip = f"coord/geoaware/{uid}_ref0.pt"
                
                try:
                    with self.coord_zip_file.open(dift_path_in_zip) as f:
                        dift_coord = torch.load(BytesIO(f.read()), map_location="cpu", weights_only=True)
                    with self.coord_zip_file.open(geoaware_path_in_zip) as f:
                        geoaware_coord = torch.load(BytesIO(f.read()), map_location="cpu", weights_only=True)
                except KeyError:
                    pass  # 文件不存在
            elif self.coord_folder:
                # 从文件夹读取
                dift_coord_path = os.path.join(self.coord_folder, "dift", f"{uid}_ref0.pt")
                geoaware_coord_path = os.path.join(self.coord_folder, "geoaware", f"{uid}_ref0.pt")
                if os.path.exists(dift_coord_path) and os.path.exists(geoaware_coord_path):
                    dift_coord = torch.load(dift_coord_path, map_location="cpu", weights_only=True)
                    geoaware_coord = torch.load(geoaware_coord_path, map_location="cpu", weights_only=True)
            
            if dift_coord is not None and geoaware_coord is not None:
                # Resize coordinates
                orig_h, orig_w = 64, 64
                new_h, new_w = self.tgt_size // 16, self.tgt_size // 16

                if (new_h, new_w) != (orig_h, orig_w):
                    # dift_coord
                    dift_coord = dift_coord.reshape(1, orig_h, orig_w, 2).permute(0, 3, 1, 2).float()
                    dift_coord = F.interpolate(dift_coord, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    dift_coord = dift_coord.permute(0, 2, 3, 1).reshape(new_h * new_w, 2).long()
                    
                    # geoaware_coord
                    geoaware_coord = geoaware_coord.reshape(1, orig_h, orig_w, 2).permute(0, 3, 1, 2).float()
                    geoaware_coord = F.interpolate(geoaware_coord, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    geoaware_coord = geoaware_coord.permute(0, 2, 3, 1).reshape(new_h * new_w, 2).long()

                ref2tgt_coords = build_ref2tgt_mapping(dift_coord, geoaware_coord, mask_tgt, tgt_patch_size=self.tgt_size // 16, ref_patch_size=self.ref_size // 16)
                coords.append(ref2tgt_coords)
            else:
                # print(f"Coord not found for uid: {uid}")
                coords.append({})
            
            caption = tgt_caption
            drop_image = False
            drop_text = False
            drop_mask = False
            rand_num = random.random()
            if "train" in self.mode:
                if rand_num < self.i_drop_rate: # 0~0.05 drop image
                    drop_image = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate): # 0.05~0.10 drop text
                    drop_text = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate): # 0.10~0.15 drop image text mask
                    drop_image = True
                    drop_text = True
                    drop_mask = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate + self.i_drop_rate): # 0.15~0.20 drop mask
                    drop_mask = True

            result = {
                'id': uid,
                'ref_imgs': [ref_img],
                'ref_masks': [mask_ref],  # 参考图像的mask
                'ref_mask_type': mask_ref_type,  # 参考图像mask的类型
                'tgt_img': tgt_img,
                'tgt_mask': mask_tgt,      # 目标图像的mask
                'tgt_mask_type': mask_tgt_type,  # 目标图像mask的类型
                'bboxs': [[0, 0, self.tgt_size, self.tgt_size]],
                'masks': [mask_tgt],       # 保持兼容性，默认使用tgt mask
                'coords': coords,
                'caption': caption,
                'drop_image': drop_image,
                'drop_text': drop_text,
                'drop_mask': drop_mask
            }

            return result

        except Exception as e:
            print(f"加载索引 {idx} 失败: {e}")
            traceback.print_exc()
            # 随机选择一个新的索引以避免无限递归
            new_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(new_index)


def make_collate_fn(num_refs: int = 6):
    """
    返回一个带参数的 collate_fn，以便 DataLoader 中简单写:
        collate_fn = make_collate_fn(num_refs=6)
    """
    def _collate(examples):
        # ---- 1. 先确定图像基础形状 (C,H,W) ----
        # 以第一条样本的 tgt_img 作为参考
        c, h, w = examples[0]["ref_imgs"][0].shape
        # ---- 2. 收集并 pad / truncate 参考图 ----
        ids = []
        ref_imgs = []
        tgt_imgs = []
        captions = []
        masks = []
        drop_images = []
        drop_texts = []
        drop_masks = []

        for ex in examples:
            refs = ex["ref_imgs"]                     # list[Tensor] 长度 0~num_refs
            refs = refs[:num_refs]                  # 若多于 num_refs 则截断

            ref_imgs.append(torch.stack(refs))      # (num_refs, C, H, W)

            ids.append(ex["id"])
            tgt_imgs.append(ex["tgt_img"])
            captions.append(ex["caption"])
            # --- 处理 masks ---
            ex_masks = ex["masks"][:num_refs]
            if len(ex_masks) < num_refs:
                # 获取目标图大小 (1, C, H, W) -> (H, W)
                _, H, W = ex["tgt_img"].shape
                pad_mask = torch.zeros((1, H, W), dtype=torch.uint8)
                ex_masks += [pad_mask] * (num_refs - len(ex_masks))
            masks.append(torch.stack(ex_masks))  # (num_refs, H, W)

            drop_images.append(ex["drop_image"])
            drop_texts.append(ex["drop_text"])
            drop_masks.append(ex["drop_mask"])

        # ---- 3. 堆 batch 维 ----
        ref_imgs = torch.stack(ref_imgs)            # (B, num_refs, C, H, W)
        tgt_imgs = torch.stack(tgt_imgs)            # (B, C, H, W)
        masks = torch.stack(masks)

        batch = {
            "ids"        : ids,
            "ref_imgs"   : ref_imgs,                # 统一 tensor
            "tgt_imgs"   : tgt_imgs,
            "captions"   : captions,                # 仍保持 list[str]
            "masks"      : masks,
            "drop_images": torch.as_tensor(drop_images, dtype=torch.bool),
            "drop_texts" : torch.as_tensor(drop_texts, dtype=torch.bool),
            "drop_masks" : torch.as_tensor(drop_masks, dtype=torch.bool),
        }
        return batch

    return _collate


def make_collate_fn_w_coord(num_refs: int = 6):
    """
    返回一个带参数的 collate_fn，以便 DataLoader 中简单写:
        collate_fn = make_collate_fn(num_refs=6)
    """
    def _collate(examples):
        # ---- 1. 先确定图像基础形状 (C,H,W) ----
        # 以第一条样本的 tgt_img 作为参考
        c, h, w = examples[0]["ref_imgs"][0].shape
        # ---- 2. 收集并 pad / truncate 参考图 ----
        ids = []
        ref_imgs = []
        ref_masks = []
        tgt_imgs = []
        tgt_masks = []
        captions = []
        masks = []
        coords = []
        drop_images = []
        drop_texts = []
        drop_masks = []

        for ex in examples:
            refs = ex["ref_imgs"]                     # list[Tensor] 长度 0~num_refs
            refs = refs[:num_refs]                  # 若多于 num_refs 则截断
            ref_imgs.append(torch.stack(refs))      # (num_refs, C, H, W)
            
            ids.append(ex["id"])
            tgt_imgs.append(ex["tgt_img"])
            captions.append(ex["caption"])
            coords.append(ex["coords"])
            
            # --- 处理 ref_masks ---
            if "ref_masks" in ex:
                ex_ref_masks = ex["ref_masks"][:num_refs]
                if len(ex_ref_masks) < num_refs:
                    # 获取ref图大小
                    _, H_ref, W_ref = refs[0].shape if len(refs) > 0 else ex["tgt_img"].shape
                    pad_mask = torch.zeros((1, H_ref, W_ref), dtype=torch.uint8)
                    ex_ref_masks += [pad_mask] * (num_refs - len(ex_ref_masks))
                ref_masks.append(torch.stack(ex_ref_masks))
            
            # --- 处理 tgt_mask ---
            if "tgt_mask" in ex:
                tgt_masks.append(ex["tgt_mask"])
            
            # --- 处理 masks (保持兼容性) ---
            ex_masks = ex["masks"][:num_refs]
            if len(ex_masks) < num_refs:
                # 获取目标图大小 (1, C, H, W) -> (H, W)
                _, H, W = ex["tgt_img"].shape
                pad_mask = torch.zeros((1, H, W), dtype=torch.uint8)
                ex_masks += [pad_mask] * (num_refs - len(ex_masks))
            masks.append(torch.stack(ex_masks))  # (num_refs, H, W)

            drop_images.append(ex["drop_image"])
            drop_texts.append(ex["drop_text"])
            drop_masks.append(ex["drop_mask"])

        # ---- 3. 堆 batch 维 ----
        ref_imgs = torch.stack(ref_imgs)            # (B, num_refs, C, H, W)
        tgt_imgs = torch.stack(tgt_imgs)            # (B, C, H, W)
        masks = torch.stack(masks)
        
        batch = {
            "ids"        : ids,
            "ref_imgs"   : ref_imgs,                # 统一 tensor
            "tgt_imgs"   : tgt_imgs,
            "captions"   : captions,                # 仍保持 list[str]
            "masks"      : masks,
            "coords"     : coords,
            "drop_images": torch.as_tensor(drop_images, dtype=torch.bool),
            "drop_texts" : torch.as_tensor(drop_texts, dtype=torch.bool),
            "drop_masks" : torch.as_tensor(drop_masks, dtype=torch.bool),
        }
        
        # 添加ref_masks和tgt_masks（如果存在）
        if ref_masks:
            batch["ref_masks"] = torch.stack(ref_masks)
        if tgt_masks:
            batch["tgt_masks"] = torch.stack(tgt_masks)
        
        return batch

    return _collate



if __name__ == "__main__":

    import os
    import random
    from datasets import load_dataset

    data_files = {"train":os.listdir("dataset/Yuanshi/Subjects200K/data")}
    dataset = load_dataset(
        "parquet", 
        data_dir="dataset/Yuanshi/Subjects200K/data", 
        data_files=data_files,
        # features=features
    )["train"]
    def filter_func(item):
        if item.get("collection") != "collection_2":
            return False
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    dataset_valid = dataset.filter(filter_func, num_proc=16)

    ref_size = 1024
    tgt_size = 1024
    # 使用自定义类
    custom_train = Subjects200K(
        original_dataset=dataset_valid,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_zip="dataset/ByteDance-FanQie/SemAlign-MS-Subjects200K/mask.zip",
        coord_zip="dataset/ByteDance-FanQie/SemAlign-MS-Subjects200K/coord.zip",
        use_rmbg_comparison=True,  # 启用RMBG mask比较
        use_ben_comparison=True    # 启用BEN mask比较
    )

    train_dataloader = torch.utils.data.DataLoader(
        custom_train, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True, 
        collate_fn=make_collate_fn_w_coord(num_refs=1),
    )

    from tqdm.auto import tqdm
    from torchvision.utils import save_image
    to_pil_image = transforms.ToPILImage()

    def patch_id_to_xy(patch_id, grid_size, image_size):
        """把 patch_id 转换成图像坐标 (x,y)"""
        y = patch_id // grid_size
        x = patch_id % grid_size
        stride = image_size / grid_size
        return int((x + 0.5) * stride), int((y + 0.5) * stride)

    def visualize_mapping_with_lines(idx, ref_img, tgt_img, ref2tgt,
                                    ref_image_size=512, tgt_image_size=1024,
                                    ref_grid_size=32, tgt_grid_size=64,
                                    max_pairs=100):
        """
        可视化 ref->tgt 的 patch 映射：在拼接图上画点并连线
        """
        ref_img = np.array(ref_img)
        tgt_img = np.array(tgt_img)

        # 把两张图拼到一起 (水平拼接)
        h = max(ref_img.shape[0], tgt_img.shape[0])
        w = ref_img.shape[1] + tgt_img.shape[1]
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas[:ref_img.shape[0], :ref_img.shape[1]] = ref_img
        canvas[:tgt_img.shape[0], ref_img.shape[1]:] = tgt_img

        pairs = []
        for r, t in ref2tgt.items():
            pairs.append((r, t))
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

        for (r, t) in pairs:
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # ref 点坐标
            rx, ry = patch_id_to_xy(r, ref_grid_size, ref_image_size)
            # tgt 点坐标（要加上 ref 图的宽度，才能映射到拼接图）
            tx, ty = patch_id_to_xy(t, tgt_grid_size, tgt_image_size)
            tx += ref_img.shape[1]

            # 画点
            cv2.circle(canvas, (rx, ry), 4, color, -1)
            cv2.circle(canvas, (tx, ty), 4, color, -1)

            # 连线
            cv2.line(canvas, (rx, ry), (tx, ty), color, 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title("Ref ↔ Tgt Patch Mapping")
        plt.savefig(f"ref2tgt_mapping_ref{idx}.png")

    # # 示例
    # for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches", total=len(train_dataloader))):
    #     ids = batch['ids']
    #     print("id: ", ids)
    #     caption = batch['captions']
    #     ref = (batch['ref_imgs'] + 1) / 2.
    #     tgt = (batch['tgt_imgs'] + 1) / 2.
    #     mask = batch['masks'].squeeze(2)
    #     coords = batch['coords'][0]
    #     from torchvision.utils import save_image
    #     for i in range(len(coords)):
    #         ref_img = ref[:, i]
    #         save_image(ref_img, f'ref_img{i}.png')
            
    #         visualize_mapping_with_lines(
    #                 i, to_pil_image(ref_img.squeeze(0)), to_pil_image(tgt.squeeze(0)), coords[i],
    #                 ref_image_size=ref_size, tgt_image_size=tgt_size,
    #                 ref_grid_size=ref_size // 16, tgt_grid_size=tgt_size // 16
    #             )

    #     save_image(tgt, 'tgt.png')
    #     save_image(mask.float(), 'mask.png')
    #     print(caption)
    # --- 随机抽取一个样本进行测试 ---
    print(f"从 {len(custom_train)} 个样本中随机抽取一个进行测试...")
    
    # 1. 获取一个随机索引并加载数据
    random_idx = 1234
    print(f"随机选择的索引: {random_idx}")
    single_item = custom_train[random_idx]

    # save_image(tgt, 'tgt.png')
    # save_image(mask.float(), 'mask.png')
    # print(caption)
    # 2. 手动 collate 单个样本，模拟 DataLoader
    collate_fn = make_collate_fn_w_coord(num_refs=1)
    batch = collate_fn([single_item])

    # 3. 执行可视化和保存逻辑
    ids = batch['ids']
    print("\n" + "="*60)
    print(f"测试样本ID: {ids[0]}")
    print("="*60)
    
    caption = batch['captions']
    ref = (batch['ref_imgs'] + 1) / 2.
    tgt = (batch['tgt_imgs'] + 1) / 2.
    mask = batch['masks'].squeeze(2)
    coords = batch['coords'][0]

    # 保存文件列表
    saved_files = []
    
    # 保存ref图像和映射可视化
    for i in range(len(coords)):
        ref_img_tensor = ref[:, i]
        ref_file = f'ref_img_{ids[0]}_{i}.png'
        save_image(ref_img_tensor, ref_file)
        saved_files.append(ref_file)
        
        mapping_file = f'ref2tgt_mapping_ref{ids[0]}_{i}.png'
        visualize_mapping_with_lines(
                f"{ids[0]}_{i}", to_pil_image(ref_img_tensor.squeeze(0)), to_pil_image(tgt.squeeze(0)), coords[i],
                ref_image_size=ref_size, tgt_image_size=tgt_size,
                ref_grid_size=ref_size // 16, tgt_grid_size=tgt_size // 16
            )
        saved_files.append(mapping_file)

    # 保存tgt图像
    tgt_file = f'tgt_{ids[0]}.png'
    save_image(tgt, tgt_file)
    saved_files.append(tgt_file)
    
    # 获取mask类型信息（后面会用到）
    tgt_mask_type = single_item.get('tgt_mask_type', 'unknown')
    ref_mask_type = single_item.get('ref_mask_type', 'unknown')
    
    print(f"\nCaption: {caption[0]}")
    print(f"\n已保存 {len(saved_files)} 个文件:")
    for idx, f in enumerate(saved_files, 1):
        print(f"  {idx}. {f}")

    # --- 4. 生成并保存所有模型的mask用于对比 ---
    print("\n" + "="*60)
    print("          生成所有模型的Mask进行对比")
    print("="*60)
    
    # 获取原始图像（未经transform的PIL图像）
    # 需要重新加载原始数据
    example = custom_train.original_dataset[random_idx]
    image = example['image']
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    
    tgt_img_pil = image.crop(
        (custom_train.padding, custom_train.padding, 
         custom_train.img_size + custom_train.padding, 
         custom_train.img_size + custom_train.padding)
    )
    ref_img_pil = image.crop(
        (custom_train.img_size + custom_train.padding * 2, custom_train.padding,
         custom_train.img_size * 2 + custom_train.padding * 2, 
         custom_train.img_size + custom_train.padding)
    )
    
    all_mask_files = []
    
    # === 处理TGT图像的所有mask ===
    print("\n[TGT图像] 生成所有mask:")
    
    # 1. 原始mask (从zip读取)
    if custom_train.grounding_zip_file:
        mask_path_in_zip = f"mask/{ids[0]}.png"
        try:
            with custom_train.grounding_zip_file.open(mask_path_in_zip) as f:
                original_mask_tgt = Image.open(BytesIO(f.read())).convert("L")
                original_mask_file = f"original_mask_tgt_{ids[0]}.png"
                original_mask_tgt.save(original_mask_file)
                all_mask_files.append(original_mask_file)
                area = get_mask_bbox_area(original_mask_tgt)
                print(f"  ✓ Original mask: {original_mask_file} (bbox面积: {area})")
        except KeyError:
            print(f"  ✗ Original mask 不存在")
    
    # 2. RMBG mask
    if custom_train.use_rmbg_comparison and custom_train.rmbg_model is not None:
        try:
            rmbg_mask_tgt = generate_rmbg_mask(
                tgt_img_pil, custom_train.rmbg_model, 
                custom_train.rmbg_transform, custom_train.device
            )
            rmbg_mask_file = f"rmbg_mask_tgt_{ids[0]}.png"
            rmbg_mask_tgt.save(rmbg_mask_file)
            all_mask_files.append(rmbg_mask_file)
            area = get_mask_bbox_area(rmbg_mask_tgt)
            print(f"  ✓ RMBG mask: {rmbg_mask_file} (bbox面积: {area})")
        except Exception as e:
            print(f"  ✗ RMBG mask 生成失败: {e}")
    
    # 3. BEN mask
    if custom_train.use_ben_comparison and custom_train.ben_model is not None:
        try:
            ben_mask_tgt = generate_ben_mask(tgt_img_pil, custom_train.ben_model)
            if ben_mask_tgt is not None:
                ben_mask_file = f"ben_mask_tgt_{ids[0]}.png"
                ben_mask_tgt.save(ben_mask_file)
                all_mask_files.append(ben_mask_file)
                area = get_mask_bbox_area(ben_mask_tgt)
                print(f"  ✓ BEN mask: {ben_mask_file} (bbox面积: {area})")
        except Exception as e:
            print(f"  ✗ BEN mask 生成失败: {e}")
    
    # === 处理REF图像的所有mask ===
    print(f"\n[REF图像] 生成所有mask (尝试ID: {ids[0]}_ref0):")
    
    # 1. 原始mask (通常不存在)
    if custom_train.grounding_zip_file:
        mask_path_in_zip = f"mask/{ids[0]}_ref0.png"
        try:
            with custom_train.grounding_zip_file.open(mask_path_in_zip) as f:
                original_mask_ref = Image.open(BytesIO(f.read())).convert("L")
                original_mask_file = f"original_mask_ref_{ids[0]}.png"
                original_mask_ref.save(original_mask_file)
                all_mask_files.append(original_mask_file)
                area = get_mask_bbox_area(original_mask_ref)
                print(f"  ✓ Original mask: {original_mask_file} (bbox面积: {area})")
        except KeyError:
            print(f"  ✗ Original mask 不存在 (正常，ref图像通常没有原始mask)")
    
    # 2. RMBG mask
    if custom_train.use_rmbg_comparison and custom_train.rmbg_model is not None:
        try:
            rmbg_mask_ref = generate_rmbg_mask(
                ref_img_pil, custom_train.rmbg_model, 
                custom_train.rmbg_transform, custom_train.device
            )
            rmbg_mask_file = f"rmbg_mask_ref_{ids[0]}.png"
            rmbg_mask_ref.save(rmbg_mask_file)
            all_mask_files.append(rmbg_mask_file)
            area = get_mask_bbox_area(rmbg_mask_ref)
            print(f"  ✓ RMBG mask: {rmbg_mask_file} (bbox面积: {area})")
        except Exception as e:
            print(f"  ✗ RMBG mask 生成失败: {e}")
    
    # 3. BEN mask
    if custom_train.use_ben_comparison and custom_train.ben_model is not None:
        try:
            ben_mask_ref = generate_ben_mask(ref_img_pil, custom_train.ben_model)
            if ben_mask_ref is not None:
                ben_mask_file = f"ben_mask_ref_{ids[0]}.png"
                ben_mask_ref.save(ben_mask_file)
                all_mask_files.append(ben_mask_file)
                area = get_mask_bbox_area(ben_mask_ref)
                print(f"  ✓ BEN mask: {ben_mask_file} (bbox面积: {area})")
        except Exception as e:
            print(f"  ✗ BEN mask 生成失败: {e}")
    
    print("\n" + "="*60)
    print(f"✓ 共生成了 {len(all_mask_files)} 个mask文件用于对比")
    print(f"✓ 数据加载时自动选择了:")
    print(f"  - TGT: [{tgt_mask_type}] mask")
    print(f"  - REF: [{ref_mask_type}] mask")
    print("="*60)

    print(f"\n✅ 测试完成！相关图片已保存，文件名以ID '{ids[0]}' 开头。")
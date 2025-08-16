# from step0_scale_smooth import scale_4_smooth
# from step1_add_white_border import 
from step2_write_text import write_text
from step3_write_number import draw_by_page_number
# from step4_add_qrcode_test import add_qrcode_test
import os
from PIL import Image
from PIL import ImageDraw, ImageFont, Image
import qrcode
import os
import torch
# import sys
import gc

Image.MAX_IMAGE_PIXELS = None  # 禁用图片大小限制

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


from PIL import Image
import numpy as np

def clear_gpu_memory():
    """清理GPU显存"""
    # 强制垃圾回收
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        print("🧹 已清理GPU显存")


def calculate_black_coverage(input_path):
    img = Image.open(input_path).convert("RGB")  # 转为 RGB
    img_np = np.array(img)

    # 定义纯白色 [255, 255, 255]
    white = np.array([255, 255, 255])

    # 计算非白色像素数量（逐像素比较 RGB）
    nonwhite_mask = np.any(img_np != white, axis=-1)  # True 表示非白
    nonwhite_pixels = np.sum(nonwhite_mask)
    total_pixels = img_np.shape[0] * img_np.shape[1]

    coverage = round(nonwhite_pixels / total_pixels, 4)

    print(f"🎨 非白色像素覆盖率: {coverage:.2%}")
    print(f"🖍️ 非白像素: {nonwhite_pixels} / 📐 总像素: {total_pixels}")
    return coverage


    
def scale_4_smooth(file_path, output_4x_path, clear_gpu_memory=True):
    # 使用绝对路径
    
    
    
    from PIL import Image
    # from realesrgan import RealESRGAN
    import torch
    import numpy as np
    import gc

    # 载入图片
    img = Image.open(file_path).convert("RGB")
    print('input img.size:', img.size)

    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    

    model = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4),
        tile=768,          # 每次只处理 512x512
        tile_pad=10,       # 防止边缘伪影
        pre_pad=0,
    )


    # 将 PIL Image 转换为 numpy 数组
    img_array = np.array(img)

    # 4倍放大
    sr_image_array, _ = model.enhance(img_array, outscale=4)

    # 将结果转换回 PIL Image
    sr_image = Image.fromarray(sr_image_array)

    sr_image_2x = sr_image.resize((int(sr_image.width//2), int(sr_image.height//2)), resample=Image.NEAREST)

    # 保存
    # sr_image.save(output_4x_path)
    return sr_image_2x
    



def scale_2_smooth(file_path, output_4x_path, clear_gpu_memory=True):
    # 使用绝对路径
    
    
    
    from PIL import Image
    # from realesrgan import RealESRGAN
    import torch
    import numpy as np
    import gc

    # 载入图片
    img = Image.open(file_path).convert("RGB")

    print(f"图片分辨率: {img.width}x{img.height}")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer


    model = RealESRGANer(
        scale=2,  # ⚠️ 改为 2，表示2倍放大
        model_path='weights/RealESRGAN_x2plus.pth',  # ⚠️ 指向你下载的 2 倍模型
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2),  # ⚠️ scale 也要改成 2
        tile=1024,
        tile_pad=10,
        pre_pad=0,
        device=torch.device('cuda')
    )



    # 将 PIL Image 转换为 numpy 数组
    img_array = np.array(img)

    # 4倍放大
    sr_image_array, _ = model.enhance(img_array, outscale=2)

    # 将结果转换回 PIL Image
    sr_image = Image.fromarray(sr_image_array)

    from PIL import ImageFilter
    sr_image = sr_image.filter(ImageFilter.SMOOTH)

    print(f"放大后图片分辨率: {sr_image.width}x{sr_image.height}")

    # 保存
    # sr_image.save(output_4x_path)
    return sr_image




def step0_scale(input_path):
    # input_path = "H:/work/doc/yuyue/图片上写字/材料/test2/1484-03罩印黑.tif"  # 输入文件名
    
    scale_4x_input_path = input_path
    scale_4x_input_path = os.path.join(SCRIPT_DIR, scale_4x_input_path)
    output_4x_path = os.path.join(SCRIPT_DIR, 'output_4x_output.png')

    # sr_image = scale_2_smooth(scale_4x_input_path, output_4x_path)
    sr_image = scale_4_smooth(scale_4x_input_path, output_4x_path)

    # scale_2x_file_path = output_4x_path
    # sr_image = Image.open(scale_2x_file_path).convert("RGB")
    print(sr_image.size)

    # # 再缩小一半实现 2 倍
    # # sr_image_2x = sr_image.resize((32030, 23220), Image.LANCZOS)
    # sr_image_2x = sr_image.resize((int(sr_image.width//2), int(sr_image.height//2)), resample=Image.NEAREST)

    sr_image = sr_image.convert("L")
    bw_image = sr_image.point(lambda x: 0 if x < 220 else 255, '1')  # mode="1"

    # scale_2x_file = 'output_final_2x_smooth_L.tif'
    scale_2x_output_path = 'step0_scale_2x_output.tif'

    output_2x_path = os.path.join(SCRIPT_DIR, scale_2x_output_path)
    # 保存
    # sr_image_2x.save(output_2x_path)
    # bw_image.save(output_2x_path, dpi=(720, 720))
    return bw_image
    

    
    # 清理中间文件以节省内存
    # if os.path.exists(output_4x_path):
    #     os.remove(output_4x_path)
    #     print("🗑️ 已删除中间文件")



def step1_add_white_border(step0_img):
    
    # input_path = "output_final_2x_smooth_L.tif"  # 输入文件名
    scale_2x_file = 'step0_scale_2x_output.tif'
    # img = Image.open(scale_2x_file)
    # img_resized = img.convert("L")

    # img_bw = img.convert("L")
    img_bw = step0_img

    final_width = 45914
    final_height = 23220
    left_margin = 6160
    background = Image.new("1", (final_width, final_height), 1)  # 白底图像
    background.paste(img_bw, (left_margin, 0))

    # === 6. 保存最终图像 ===
    output_path = "step1_add_white_border_output.tif"
    # background.save(output_path, dpi=(720, 720))

    step1_img = background
    return step1_img


def step2_write_text(text_to_write, step1_img):
        
    # 文件路径
    # input_path = "output_final.tif"
    # input_path = "step1_add_white_border_output.tif"
    output_path = "step2_add_text_output.tif"

    # 打开图像为灰度图 (L 模式)
    # img = Image.open(input_path).convert("L")
    img = step1_img
    draw = ImageDraw.Draw(img)

    # # 加载字体
    # try:
    #     font = ImageFont.truetype("simkai.ttf", 173)
    # except:
    #     print("⚠️ 没找到楷体，使用默认字体")
    #     font = ImageFont.load_default()

    # === 写竖排文字（每个字旋转90度）===
    # text = "C24 820 1130 24011484-12-03 印黑 135B JY YY24002# TT25655 2024-1-29 覆盖率0.627%"
    x_center = 3958
    start_y = 1773
    line_spacing = 260

    overlap_adjust = 10  # 减少字符间距，贴紧一点



    # 设定目标宽高（以6mm × N字符为准）
    char_size_mm = 6
    dpi = 720
    char_px = int(char_size_mm / 25.4 * dpi)  # ≈170px

    char_px += 20
    print('char_px:', char_px)

    # 待写的文本
    # text = "C24 820 1130 24011484-12-03 印黑 135B JY YY24002# TT25655 2024-1-29 覆盖率0.6%"
    text = text_to_write

    # 估算整体图像尺寸：长为 N 字 × 每字像素宽度，宽为1个字宽度
    width = len(text) * char_px
    height = char_px

    # 创建横向文字图像（L灰度）
    text_img = Image.new("L", (width, height), color=255)
    text_draw = ImageDraw.Draw(text_img)

    # 加载字体
    try:
        font = ImageFont.truetype("simkai.ttf", char_px)
    except:
        print("⚠️ 未找到 simkai.ttf，使用默认字体")
        font = ImageFont.load_default()

    # 写文字（加粗实现：轻微多次偏移）
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            text_draw.text((0 + dx, 0 + dy), text, font=font, fill=0)

    # 旋转整个文本图像（竖排显示）
    rotated_img = text_img.rotate(-90, expand=True)

    # 粘贴到主图上
    img.paste(rotated_img, (x_center, start_y))



    # === 横线1 ===
    draw.line((0, 170, 270, 170), fill=0, width=8)

    # === 加号1 ===
    cx, cy = 2345, 170
    draw.line((cx - 68, cy, cx + 68, cy), fill=0, width=16)
    draw.line((cx, cy - 74, cx, cy + 74), fill=0, width=16)

    # === 加号2 ===
    cx, cy = 4410, 170
    draw.line((cx - 158, cy, cx + 158, cy), fill=0, width=16)
    draw.line((cx, cy - 155, cx, cy + 155), fill=0, width=16)

    # === 加号3 ===
    cx, cy = 41504, 170
    draw.line((cx - 158, cy, cx + 158, cy), fill=0, width=16)
    draw.line((cx, cy - 155, cx, cy + 155), fill=0, width=16)

    # === 加号4 ===
    cx, cy = 43562, 170
    draw.line((cx - 68, cy, cx + 68, cy), fill=0, width=16)
    draw.line((cx, cy - 74, cx, cy + 74), fill=0, width=16)

    # === 横线2 ===
    draw.line((45914, 170, 45914 - 270, 170), fill=0, width=8)



    # === 转为1-bit图像并保存 ===
    bw_img = img.convert("1")
    # group4 是无损压缩
    # bw_img.save(output_path, dpi=(720, 720), compression="group4")
    # bw_img.save(output_path, dpi=(720, 720))

    print("✅ 已完成所有图形绘制并保存为：", output_path)
    step2_img = bw_img
    return step2_img


def step3_write_number(total_page_number, page_number, step2_img):
        
    
    # input_path = "step2_add_text_output.tif"
    # img = Image.open(input_path).convert("L")
    img = step2_img
    output_path = "step3_add_number_test_output.tif"
    # font_path = "simkai.ttf"
    font_path = "C:/Windows/Fonts/Toothpick.ttf"
    # page_number = 1
    # page_number = 2
    # total_page_number = 5

    # === 圆圈：纵向排列，从(4812, 6358) 开始 ===
    outer_diameter = 298
    inner_diameter = 238
    circle_spacing = 30
    x0, y0 = 4812, 6358

    output_path_template = "step3_add_number_{}.tif"

    # for page_number in range(1, total_page_number + 1):
    #     output_path = output_path_template.format(page_number)
    #     draw_by_page_number(input_path, output_path, page_number, total_page_number, font_path,
    #                         outer_diameter=outer_diameter, inner_diameter=inner_diameter, circle_spacing=circle_spacing, x0=x0, y0=y0)
    output_path = output_path_template.format(page_number)
    step3_img = draw_by_page_number(img, output_path, page_number, total_page_number, font_path,
                            outer_diameter=outer_diameter, inner_diameter=inner_diameter, circle_spacing=circle_spacing, x0=x0, y0=y0)
    return step3_img




def step4_add_qrcode(step3_img, output_path):
    
    # 你要在扫码后显示的文字
    text_to_encode = "https://imgcdnv1.fabricschina.com.cn/chat-images/generated/1751506547013.png?basic=30p"

    # 生成二维码
    # qr = qrcode.QRCode(box_size=100, border=2)
    # qr.add_data(text_to_encode)
    # qr.make(fit=True)
    # qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")


    # 生成二维码
    qr = qrcode.QRCode(
        version=None,        # 自动调整二维码复杂度
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,         # 单个模块像素大小（会被覆盖）
        border=4             # 边框宽度（单位是 box）
    )
    qr.add_data(text_to_encode)
    qr.make(fit=True)

    # 得到二维码图像（初始大小）
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    # 调整为 710x710 尺寸
    resized_img = img.resize((710, 710), resample=Image.LANCZOS)

    # for input_path, output_path in zip(input_paths, output_paths):

        # 打开原始图片
    # base_image = Image.open(input_path)  # 换成你的原图路径
    base_image = step3_img

    # 设定粘贴坐标（左上角）例如：右下角 100x100 范围
    # x = base_image.width - qr_img.width - 50
    # y = base_image.height - qr_img.height - 50

    x = 4045
    y = 770

    # 粘贴二维码到原图
    base_image.paste(resized_img, (x, y))

    compression = "tiff_lzw"

    original_dpi = (720, 720)
    save_params = {
        "format": "TIFF",
        "compression": compression,  # None（无压缩），或 "tiff_lzw"
        "dpi": original_dpi
    }

    # 保存结果
    base_image.save(output_path, **save_params)




if __name__ == "__main__":
    input_paths = []
    input_paths.append("H:/work/doc/yuyue/图片上写字/材料/（未组版文件）1484-01深兰字边.pcx")
    input_paths.append("H:/work/doc/yuyue/图片上写字/材料/（未组版文件）1484-02防浅兰布边.pcx")
    input_paths.append("H:/work/doc/yuyue/图片上写字/材料/test2/1484-03罩印黑.tif")
    input_paths.append("H:/work/doc/yuyue/图片上写字/材料/test2/1484-04罩印黑花.tif")
    pre_output_paths = []
    pre_output_paths.append("1484-01深兰字边")
    pre_output_paths.append("1484-02防浅兰布边")
    pre_output_paths.append("1484-03罩印黑")
    pre_output_paths.append("1484-04罩印黑花")
    text_to_writes = []
    text_to_writes.append("C24 820 1130 24011484-12-01 深兰字边 80C JY YY24002# TT25655 2025-7-12 覆盖率:{}")
    text_to_writes.append("C24 820 1130 24011484-12-02 防浅兰布边 80C  JY YY24002# TT25655 2025-7-12 覆盖率：{}")
    text_to_writes.append("C24 820 1130 24011484-12-03 罩印黑 135B JY YY24002# TT25655 2025-7-12  覆盖率：{}")
    text_to_writes.append("C24 820 1130 24011484-12-04 罩印黑花 125B JY YY24002# TT25655 2025-7-12  覆盖率:{}")
    cover_rates = []
    for i in input_paths:
        input_path = i
        pre_output_path = i.split(".")[0]
        cover_rate = calculate_black_coverage(input_path)
        print(f"🧮 黑色覆盖率（原图）: {cover_rate:.4%}")
        cover_rates.append(f"{cover_rate:.2%}")
    
    index = 1
    for input_path, pre_output_path, text_to_write, cover_rate in zip(input_paths, pre_output_paths, text_to_writes, cover_rates):
        # if index != 2:
        #     index += 1
        #     continue
        print(f"\n🔄 开始处理第 {index} 个文件: {input_path}")
        
        # 在开始处理前清理显存
        clear_gpu_memory()
        
        # input_path = "H:/work/doc/yuyue/图片上写字/材料/（未组版文件）1484-01深兰字边.pcx"  # 输入文件名
        # pre_output_path = "1484-01深兰字边"
        # # input_path = "H:/work/doc/yuyue/图片上写字/材料/test2/1484-03罩印黑.tif"  # 输入文件名
        # cover_rate = calculate_black_coverage(input_path)
        # print(f"🧮 黑色覆盖率（原图）: {cover_rate:.4%}")
        # text_to_write = f"C24 820 1130 24011484-12-01 深兰字边 80C JY YY24002# TT25655 2025-7-12 覆盖率:{cover_rate:.2%}"
        # print(text_to_write)
        text_to_write = text_to_write.format(cover_rate)

        try:
            import time
            start_time = time.time()
            
            step0_img = step0_scale(input_path)
            print(f"⏱️ step0_scale 耗时: {time.time() - start_time:.2f}秒")
            step1_start = time.time()
            
            step1_img = step1_add_white_border(step0_img)
            print(f"⏱️ step1_add_white_border 耗时: {time.time() - step1_start:.2f}秒")
            step2_start = time.time()
            
            step2_img = step2_write_text(text_to_write, step1_img)
            print(f"⏱️ step2_write_text 耗时: {time.time() - step2_start:.2f}秒")
            step3_start = time.time()
            
            total_page_number = 4
            step3_img = step3_write_number(total_page_number, index, step2_img)
            print(f"⏱️ step3_write_number 耗时: {time.time() - step3_start:.2f}秒")
            step4_start = time.time()
            
            # input_path = f"step3_add_number_{index}.tif" 
            output_path = f"{pre_output_path}_{index}.tif" 
            step4_img = step4_add_qrcode(step3_img, output_path)
            print(f"⏱️ step4_add_qrcode 耗时: {time.time() - step4_start:.2f}秒")
            
            print(f"⏱️ 总耗时: {time.time() - start_time:.2f}秒")
            
            print(f"✅ 第 {index} 个文件处理完成")
            
        except Exception as e:
            print(f"❌ 处理第 {index} 个文件时出错: {e}")
        finally:
            # 每个循环结束后清理显存
            clear_gpu_memory()
            
        index += 1











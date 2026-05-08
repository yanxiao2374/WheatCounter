import os
import shutil
src_folders=[r"/data1/mazc/mrf/Datasets/TransCrowd/GlobalWheat2020/resize_images/train",]

# 目标文件夹（移动到这里）
dst_folder = r"/data1/mazc/mrf/Datasets/TransCrowd/GlobalWheat2020/foreground/train"

# 要匹配的结尾数字
suffixes = ('11', '20', '21')

# 遍历源文件夹
for src_folder in src_folders:
    for filename in os.listdir(src_folder):
        # 检查是否是图片文件
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            # 去掉扩展名再判断是否以11/20/21结尾
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext.endswith(suffixes):
                src_path = os.path.join(src_folder, filename)
                dst_path = os.path.join(dst_folder, filename)

                # 移动并覆盖
                shutil.copy2(src_path, dst_path)
                print(f"已移动并覆盖: {filename}")

print("操作完成！")
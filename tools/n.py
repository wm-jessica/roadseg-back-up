from PIL import Image
import os
from osgeo import gdal
import numpy as np

def create_geotiff_from_png(png_path, tiff_path, original_tiff_path, output_tiff_path):
    # 读取原始TIFF图像的地理信息和仿射矩阵
    original_proj, original_geotrans, _ = read_img(original_tiff_path)

    # 读取PNG图像
    png_image = Image.open(png_path)
    png_data = np.array(png_image)

    # 创建新的TIFF文件
    driver = gdal.GetDriverByName("GTiff")
    png_width, png_height = png_data.shape[1], png_data.shape[0]
    dataset = driver.Create(output_tiff_path, png_width, png_height, 3, gdal.GDT_Byte)

    # 设置仿射变换参数和地图投影信息
    dataset.SetGeoTransform(original_geotrans)
    dataset.SetProjection(original_proj)

    # 写入PNG数据到TIFF文件的RGB通道
    for i in range(3):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(png_data[:, :, i])

    # 保存TIFF文件
    dataset = None

if __name__ == '__main__':
    # PNG图像路径
    png_path = "path_to_predicted_image.png"
    
    # 原始TIFF图像路径（用于获取地理信息和仿射矩阵）
    original_tiff_path = "path_to_original_tiff_image.tif"
    
    # 输出的TIFF图像路径
    output_tiff_path = "path_to_output_tiff_image.tif"
    
    create_geotiff_from_png(png_path, output_tiff_path, original_tiff_path)

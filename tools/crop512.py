import os
from osgeo import gdal



def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    return im_proj, im_geotrans, im_data


def write_img(filename, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

# 有两个文件夹bj-5 bj-dx
# 每个文件夹下有4副影像
# 遍历每个影像
# 按需修改
#area_list = ['bj-5','bj-dx']
#index_list = ['111','02','03','04']
#area_list = ['GFimage']
#index_list = ['GFimagetraindata']
area_list = ['GF2-final']
index_list = ['20191121_L1A0004409303', '20191121_L1A0004409978', '20191121_L1A0004409979',  '20200402_L1A0004714112', '20200402_L1A0004714119']
for m in area_list:
    for k in index_list:
        #图片的路径
        p = "D:/2020traindata/GF-2/GF-2data/" + m + "/" +  k +".tif"
        #读取图片
        im_proj,im_geotrans,im_data = read_img(p)
        channel,width,height = im_data.shape#GFimage获取图片的宽、高、通道
        #width,height = im_data.shape#labelimage获取图片的宽、高
        for i in range(width//512):       #451修改为自己需要的宽
         for j in range(height//512):     #451修改为自己需要的高
                 cur_img = im_data[:,i*512:(i+1)*512,j*512:(j+1)*512]#GFimage
                 #cur_img = im_data[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]#labelimage
                 #保存图片的命名格式（可以自己修改
                 write_img('D:/2020traindata/cuttiff/{}/{}/{}_{}_{}.tif'.format(m,k,k,i,j), im_geotrans, cur_img)


# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
from os import listdir, getcwd
from os.path import join
 
 
sets = ['train', 'test', 'val']
classes= ['weed','crop']
 
for image_set in sets:
#     '''
#     对所有的文件数据集进行遍历
#     做了两个工作：
# 　　　　１．将所有图片文件都遍历一遍，并且将其所有的全路径都写在对应的txt文件中去，方便定位
# 　　　　２．同时对所有的图片文件进行解析和转化，将其对应的bundingbox 以及类别的信息全部解析写到label 文件中去
#     　　　　　最后再通过直接读取文件，就能找到对应的label 信息
#     '''
    # 先找labels文件夹如果不存在则创建
    if not os.path.exists('data/data_weed/labels/'):
        os.makedirs('data/data_weed/labels/')
    # 读取在ImageSets/Main 中的train、test..等文件的内容
    # 包含对应的文件名称
    image_ids = open('data/data_weed/ImageSets/%s.txt' % (image_set)).read().strip().split()
    # 打开对应的2012_train.txt 文件对其进行写入准备
    
    # 将对应的文件_id以及全路径写进去并换行
    if not os.path.exists('data/data_weed/dataSet_path/'):
        os.makedirs('data/data_weed/dataSet_path/')
    list_file = open('data/data_weed/dataSet_path/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
       
        list_file.write('/home/robint01/yolov5/data/data_weed/images/%s.jpg\n' % (image_id))
        # convert_annotation(image_id)
        # list_file.write('data/data_weed/images/%s.jpg\n' % (image_id))
        # 调用  year = 年份  image_id = 对应的文件名_id
        # try:
        #     convert_annotation(image_id)
        # except:
        #     continue
    # 关闭文件
    list_file.close()
 
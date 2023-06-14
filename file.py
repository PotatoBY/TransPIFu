import os
import glob
import shutil

path_photo = '/mnt/data/Project/Geo-PIFu-master/data/humanRender/geopifuResults/*.png'#所有photo所在的文件夹目录
#
files_list = glob.glob(path_photo) # 得到文件夹下的所有文件名称，存在字符串列表中
for path in files_list:
    shutil.move(path, '/mnt/data/Project/Geo-PIFu-master/data/humanRender/geopifuResults/new')

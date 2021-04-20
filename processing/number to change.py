import os
#文件路径，注意用/或\\,不能用\
path='pp'
#返回指定的文件夹包含的文件或文件夹的名字的列表
file_list=os.listdir(path)
n=1
for file_obj in file_list:
    #针对某一种文件，比如.jpg文件
    if file_obj.endswith('.jpg'):
        #之前的文件名
        src=os.path.join(path,file_obj)
        #根据自己的需要设置新文件名。format中{:0>5d}的含义：数字补零 (填充左边, 宽度为5)
        newname =  '{:0>5d}.jpg'.format(n)
        dst = os.path.join(path, newname)
        #用os模块中的rename方法对文件改名
        os.rename(src,dst)
        print(src,'======>',dst)
    n+=1

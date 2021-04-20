
from PIL import Image
import os 
 
images_dir = 'C:/Users/Administrator/Desktop/p/'
count = os.listdir(images_dir)
print("count=",len(count))
print("len(count)+1=",len(count)+1)
for i in range(1,len(count)+1):
    im = Image.open(images_dir+str(i).zfill(5)+'.jpg')
    im_size = im.size
    print("图片宽度和高度分别是{}".format(im_size))
    if(im_size[0]>im_size[1]):
        print(im_size[0],im_size[1])
        im = im.resize((480,300))
        im.save(images_dir+str(i).zfill(5)+'.jpg')
    else:
        im = im.resize((300,480))
        im.save(images_dir+str(i).zfill(5)+'.jpg')

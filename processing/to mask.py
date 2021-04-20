import os
path='labelme_json'
files=os.listdir(path)
for file in files:
    jpath=os.listdir(os.path.join(path,file))
#     print(file[:-5])
    new=file[:-5]
#     print(jpath[0])
#     newname=os.path.join(path,file,new)
    newnames=os.path.join('cv2_mask',new)
    filename=os.path.join(path,file,jpath[0])
    print(filename)
    print(newnames)
    os.rename(filename,newnames+'.png')


#Extract all pictures in the directory, change the size and save to another directory
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=580,height=838):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("p\\*.jpg"):     # Images files address
    convertjpg(jpgfile,"pp")             # Save address
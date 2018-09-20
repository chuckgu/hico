import pickle
# output = open('hico.pkl', 'rb')
# train=pickle.load(output)
# output.close()
#
# output = open('hico_test.pkl', 'rb')
# test=pickle.load(output)
# output.close()
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

resnet50 = models.resnet50(pretrained=True)
modules=list(resnet50.children())[:-1]
resnet50=nn.Sequential(*modules)

resnet50.eval()


def save_feature(data,mode):
    output = open('obj_'+mode+'.pkl', 'rb')
    data = pickle.load(output)
    output.close()

    for d in data:

        obj=an["obj"]
        obj_id=an["obj_id"]
        img_path=an["obj_file"]


        print(img_path)

        jpgfile = Image.open("./hico/images/"+mode+"/"+img_path)

        bbox=d["bbox"][0]

        x1, x2, y1, y2=bbox["obj_box"]

        hoi=bbox["hoi"]

        obj=hoi[0]

        obj_id=obj2id[obj]

        ojb_file=jpgfile.crop((x1,y1,x2,y2))

        ojb_file.save("./"+mode+"_obj/"+img_path)

        an={}
        an["obj"]=obj
        an["obj_id"]=obj_id
        an["obj_file"]=img_path

        anno.append(an)

    output = open('obj_'+mode+'.pkl', 'wb')
    pickle.dump(anno, output)
    output.close()

    print("saved")










print("done")
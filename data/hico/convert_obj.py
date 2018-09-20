import pickle
output = open('hico.pkl', 'rb')
train=pickle.load(output)
output.close()

output = open('hico_test.pkl', 'rb')
test=pickle.load(output)
output.close()
from PIL import Image


def convert(data,mode):
    obj2id=data["obj2id"]
    verb2id = data["verb2id"]

    anno=[]

    for d in data["anno"]:
        img_path=d["img_path"]
        print(img_path)

        jpgfile = Image.open("./hico/images/"+mode+"/"+img_path)

        bbox=d["bbox"][0]

        x1, x2, y1, y2=bbox["obj_box"]

        hoi=bbox["hoi"]

        obj=hoi[0]

        obj_id=obj2id[obj]

        ojb_file=jpgfile.crop((x1,y1,x2,y2))

        ojb_file.save("./"+mode+"_obj/"+img_path)

        verb=hoi[1]

        verb_id=verb2id[verb]

        an={}
        an["obj"]=obj
        an["obj_id"]=obj_id
        an["obj_file"]=img_path

        an["verb"]=verb
        an["verb_id"]=verb_id

        anno.append(an)

    output = open('anno_'+mode+'.pkl', 'wb')
    pickle.dump(anno, output)
    output.close()

    print("saved")




convert(train,"train")
convert(test,"test")








print("done")
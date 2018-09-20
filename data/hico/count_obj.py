import pickle,cv2
from scipy import io
import operator
import numpy as np
output = open('hico.pkl', 'rb')
train=pickle.load(output)
output.close()

output = open('hico_test.pkl', 'rb')
test=pickle.load(output)
output.close()
from PIL import Image

loaded = io.loadmat('anno_bbox.mat')

list_action = {}
for i, hoi in enumerate(loaded['list_action']):
    # action = {}
    obj = hoi[0][0][0]
    verb = hoi[0][1][0]
    verb_ing =hoi[0][2][0]
    list_action[i + 1] = (obj, verb, verb_ing)


def convert(data,mode):
    obj2id=data["obj2id"]
    verb2id = data["verb2id"]

    anno=[]

    count_ids={}

    for i,d in enumerate(data["anno"]):
        img_path=d["img_path"]
        print(img_path)


        bbox=d["bbox"]

        for j,box in enumerate(bbox):

            ids=box["id"]

            count_ids[ids] = count_ids.get(ids, 0) + 1
            #
            # x1, x2, y1, y2=box["obj_box"]
            #
            # pt=(x1,y1,x2,y2)
            #
            # hoi=box["hoi"]
            #
            # obj=hoi[0]
            #
            # obj_id=obj2id[obj]
            #
            #
            #
            # x1, x2, y1, y2 = box["human_box"]
            #
            # pt2 = (x1, y1, x2, y2)
            #
            #
            #
            # verb = hoi[1]
            #
            # verb_id = verb2id[verb]
            #
            #
            # coords_1 = ((pt[0] + pt[2]) / 2, (pt[1] + pt[3]) / 2)
            # coords_2 = ((pt2[0] + pt2[2]) / 2, (pt2[1] + pt2[3]) / 2)









        # an={}
        # an["obj"]=obj
        # an["obj_id"]=obj_id
        # an["obj_file"]=img_path
        #
        # an["verb"]=verb
        # an["verb_id"]=verb_id
        #
        # anno.append(an)

    sorted_x = sorted(count_ids.items(), key=operator.itemgetter(1),reverse=True)

    final_list=[]

    for x in sorted_x:
        ids, cnt=x
        obj, verb, verb_ing=list_action[ids]

        final_list.append([verb,obj,cnt])

    np.savetxt("counts_test.csv", np.asarray(final_list), delimiter=",", fmt="%s")





    print("saved")




convert(test,"test")








print("done")
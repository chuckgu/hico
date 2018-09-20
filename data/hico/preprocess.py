from scipy import io
import numpy as np

loaded = io.loadmat('anno_bbox.mat')

lines = [line.rstrip('\n') for line in open('list_obj.txt')]

id2obj = {int(l.split()[0]): l.split()[1] for l in lines}

obj2id = {l.split()[1]: int(l.split()[0]) for l in lines}

lines = [line.rstrip('\n') for line in open('list_vb.txt')]

id2verb = {int(l.split()[0]): l.split()[1] for l in lines}

verb2id = {l.split()[1]: int(l.split()[0]) for l in lines}

def preprocess(mode="train"):
    list_action = {}
    for i, hoi in enumerate(loaded['list_action']):
        # action = {}
        obj = hoi[0][0][0]
        verb = hoi[0][1][0]
        verb_ing =hoi[0][2][0]
        list_action[i + 1] = (obj, verb, verb_ing)
        # list_action.append(action)

    anno = []
    for img in loaded['bbox_'+mode][0]:
        data = {}

        img_path = img[0][0]
        width = img[1][0][0][0][0][0]
        height = img[1][0][0][1][0][0]

        data["img_path"] = img_path
        data["width"] = width
        data["height"] = height

        bbox_list = []

        print(img_path)

        for i in range(len(img[2][0])):
            box = {}

            bbox = img[2][0][i]
            ids = bbox[0][0][0]
            box["id"] = ids

            if len(bbox[1])==0:
                continue
            # print(ids)
            hoi=list_action[ids]
            box["hoi"]=hoi

            x1 = bbox[1][0][0][0][0][0]
            x2 = bbox[1][0][0][1][0][0]
            y1 = bbox[1][0][0][2][0][0]
            y2 = bbox[1][0][0][3][0][0]

            box["human_box"] = (x1, y1, x2, y2)
            box["human_class"] = verb2id[hoi[1]]


            h_x1 = bbox[2][0][0][0][0][0]
            h_x2 = bbox[2][0][0][1][0][0]
            h_y1 = bbox[2][0][0][2][0][0]
            h_y2 = bbox[2][0][0][3][0][0]

            box["obj_box"] = (h_x1, h_y1, h_x2, h_y2)
            box["obj_class"] = obj2id[hoi[0]]

            if h_x2<h_x1:
                print(box["obj_box"])

            con = bbox[3][0][0]
            inv = bbox[3][0][1]

            box["connection"] = con
            box["invis"] = inv

            bbox_list.append(box)

        data["bbox"] = bbox_list

        anno.append(data)

    anno=[a for a in anno if len(a['bbox'])>0]


    savedict={"anno":anno,
              "hoi":list_action,
              "id2obj":id2obj,
              "obj2id":obj2id,
              "id2verb":id2verb,
              "verb2id":verb2id}


    import pickle
    output = open("../hico_"+mode+".pkl", 'wb')
    pickle.dump(savedict, output)
    output.close()


    print("done")


preprocess()
preprocess("test")








import matplotlib.pyplot as plt
import numpy as np
import math
import os
import shutil
import cv2
import sys
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_image_path="./dogs-vs-cats/train/"
test_image_path="./dogs-vs-cats/test1/"
train_image_list=[]
test_image_list=[]
rows = 299
cols = 299

#从目录中所有文件读入到列表中
def get_image_list(path_name, list_name):
    for file_name in os.listdir(path_name):
        list_name.append(os.path.join(path_name, file_name))

get_image_list(train_image_path, train_image_list)
get_image_list(test_image_path, test_image_list)
print("train image sample:{}\ntest image sample:{}".format(len(train_image_list),len(test_image_list)))

def display_img(img_list, summary = True):
    fig = plt.figure(figsize=(15, 2 * math.ceil(len(img_list)/8)))
    for i in range(0, len(img_list)):
        img = cv2.imread(img_list[i])
        img = img[:,:,::-1]#BGR->RGB
        if summary:
            print("---->image: {}  - shape: {}".format(img_list[i], img.shape))
        ax = fig.add_subplot(math.ceil(len(img_list)/8),8,i+1)
        ax.set_title(os.path.basename(img_list[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        img = cv2.resize(img, (128,128))
        ax.imshow(img)
    plt.show()

# import random
# random.seed(2018)
# display_img(random.sample(train_image_list, 10))
# display_img(random.sample(test_image_list, 10))
#
# #实现获取图片像素函数
# def get_pic_size_distribution(img_list):
#     x_PX= np.zeros(25000)
#     y_PX= np.zeros(25000)
#     for i,item in enumerate(img_list):
#         img = cv2.imread(item)
#         x_PX[i]=img.shape[0]
#         y_PX[i]=img.shape[1]
#     return x_PX, y_PX
#
# #实现展示图片size分布图函数
# def show_pic_size_distribution( x_PX, y_PX ):
#     plt.figure(figsize=(15,15))
#     #设置lable，颜色
#     plt.scatter(x_PX, y_PX, c='blue', label='px')
#     #设置标题
#     plt.title('pic_size_distribution')
#     #设置坐标轴lable
#     plt.xlabel('x_px')
#     plt.ylabel('y_px')
#     #设置legend
#     plt.legend(loc=2)
#     plt.show()
#
# #展示训练集图片size分布
# x_PX, y_PX = get_pic_size_distribution(train_image_list)
# show_pic_size_distribution( x_PX, y_PX )
#
# abnormal=[]
# for i in range(25000):
#     if y_PX[i]>800:
#         abnormal.append(train_image_list[i])
#
# display_img(abnormal)

model_pre=keras.applications.Xception(weights='imagenet')
Dogs = ['n02085620','n02085782','n02085936','n02086079','n02086240','n02086646','n02086910','n02087046','n02087394','n02088094','n02088238',
        'n02088364','n02088466','n02088632','n02089078','n02089867','n02089973','n02090379','n02090622','n02090721','n02091032','n02091134',
        'n02091244','n02091467','n02091635','n02091831','n02092002','n02092339','n02093256','n02093428','n02093647','n02093754','n02093859',
        'n02093991','n02094114','n02094258','n02094433','n02095314','n02095570','n02095889','n02096051','n02096177','n02096294','n02096437',
        'n02096585','n02097047','n02097130','n02097209','n02097298','n02097474','n02097658','n02098105','n02098286','n02098413','n02099267',
        'n02099429','n02099601','n02099712','n02099849','n02100236','n02100583','n02100735','n02100877','n02101006','n02101388','n02101556',
        'n02102040','n02102177','n02102318','n02102480','n02102973','n02104029','n02104365','n02105056','n02105162','n02105251','n02105412',
        'n02105505','n02105641','n02105855','n02106030','n02106166','n02106382','n02106550','n02106662','n02107142','n02107312','n02107574',
        'n02107683','n02107908','n02108000','n02108089','n02108422','n02108551','n02108915','n02109047','n02109525','n02109961','n02110063',
        'n02110185','n02110341','n02110627','n02110806','n02110958','n02111129','n02111277','n02111500','n02111889','n02112018','n02112137',
        'n02112350','n02112706','n02113023','n02113186','n02113624','n02113712','n02113799','n02113978']
Cats=['n02123045','n02123159','n02123394','n02123597','n02124075','n02125311','n02127052']


def batch_img(img_path_list, batch_size):
    '''split img_path_list into batches'''
    for begin in range(0, len(img_path_list), batch_size):
        end = min(begin + batch_size, len(img_path_list))
        yield img_path_list[begin:end]


def read_batch_img(batch_imgpath_list):
    '''read batch img and resize'''
    images = np.zeros((len(batch_imgpath_list), 299, 299, 3), dtype=np.uint8)
    for i in range(len(batch_imgpath_list)):
        img = cv2.imread(batch_imgpath_list[i])
        img = img[:, :, ::-1]
        img = cv2.resize(img, (299, 299))
        images[i] = img
    return images


def pred_pet(model, img_path_list, top_num, preprocess_input, decode_predictions, batch_size=32):
    '''predict img
    #returns
        the list, will show pet or not
    '''
    ret = []
    for batch_imgpath_list in batch_img(img_path_list, batch_size):
        print(len(ret))
        X = read_batch_img(batch_imgpath_list)
        X = preprocess_input(X)
        preds = model.predict(X)
        dps = decode_predictions(preds, top=top_num)
        for index in range(len(dps)):
            for i, val in enumerate(dps[index]):
                if (val[0] in Dogs) and ('dog' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                elif (val[0] in Cats) and ('cat' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                if i == len(dps[index]) - 1:
                    ret.append(False)
    return ret

def get_abnormal_v(train_image_list, topN = 50):
    abnormal_v = []
    if os.path.exists("./abnormal.txt"):
        with open("./abnormal.txt", 'r') as f:
            items = f.readlines()
            abnormal_v = [item.strip('\n') for item in items]
    else:
        ret =[]
        ret = pred_pet(model_pre, train_image_list, topN, keras.applications.xception.preprocess_input, keras.applications.xception.decode_predictions)
        for i,val in enumerate(ret):
            if not val:
                abnormal_v.append(train_image_list[i])
        with open("./abnormal.txt", 'w') as f:
            for item in abnormal_v:
                f.write("{}\n".format(item))
    return abnormal_v

# abnormal_v = get_abnormal_v(train_image_list, topN=20)
# display_img(abnormal_v, summary = True)
# train_image_list = [item for item in train_image_list if item not in abnormal_v]
# for i in abnormal_v:
#     os.remove(i)

train2_file_cat = ['./train2/cat/' + img for img in os.listdir('./train2/cat/')]
train2_file_dog = ['./train2/dog/' + img for img in os.listdir('./train2/dog/')]

validation_file_cat = ['./validation/cat/' + img for img in os.listdir('./validation/cat/')]
validation_file_dog = ['./validation/dog/' + img for img in os.listdir('./validation/dog/')]

print(len(train2_file_cat))
print(len(train2_file_dog))
print(len(validation_file_cat))
print(len(validation_file_dog))
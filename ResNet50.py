import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import matplotlib.pyplot as plt
import numpy as np


ROWS = 299
COLS = 299
CHANNELS = 3
batch_size = 32
epochs = 10


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dir = './train2'
validation_dir = './validation'
test_dir = './test1'

Inp = keras.layers.Input((ROWS, COLS, CHANNELS))
ResNet50_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, CHANNELS))
x = ResNet50_model(Inp)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
output = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs=Inp, outputs=output)

for layer in ResNet50_model.layers:
    layer.trainable = False
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='ResNet50_model.pdf')

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 20,       # 随机错切变换
    zoom_range = 0.2,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    fill_mode = 'nearest',   # 填充方式
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
)


# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(ROWS,COLS),
    batch_size=batch_size,
    )

# 验证数据
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(ROWS,COLS),
    batch_size=batch_size,
    )

model.summary()

# 定义优化器，代价函数，训练过程中计算准确率
model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs = epochs,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator)
                    )

model.save('ResNet50.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc = [
# 0.8516348705653,
# 0.8438235441390015,
# 0.8677081768564419,
# 0.9364077912988735,
# 0.9474738370637424,
# 0.9512292824595664,
# 0.9510790646437334,
# 0.9567873416453858,
# 0.9587401732541988,
# 0.9579390115667719]
# val_acc = [
# 0.585888955666704,
# 0.7282020444620537,
# 0.7434355582997851,
# 0.956905191445021,
# 0.9625175385848868,
# 0.9669272399278412,
# 0.9711365001931366,
# 0.9723391460139423,
# 0.9703347363125994,
# 0.9697334134021964]
# loss = [
# 0.3716633921840458,
# 0.3741126062634723,
# 0.34369935425551756,
# 0.17997929030788434,
# 0.14715111184706495,
# 0.13448975421572226,
# 0.12611873927518483,
# 0.11550908803098174,
# 0.11162807294861722,
# 0.10750037845153072]
# val_loss = [
# 1.0478995381579128,
# 0.5521533883843734,
# 0.5073287922266992,
# 0.11485915799959553,
# 0.09867928910154478,
# 0.08831558105050263,
# 0.07956882225867383,
# 0.07733541514554992,
# 0.08139585443311474,
# 0.08032926734209085]

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('ResNet50_result.pdf')
from matplotlib.ticker import MultipleLocator

# 绘制训练 & 验证的准确率值
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(acc, color='blue', linestyle='-', label='Train accuracy')
lns2 = ax.plot(val_acc, color='orange', linestyle='-', label='Validation accuracy')
ax2 = ax.twinx()
lns3 = ax2.plot(loss, color='red', linestyle='-', label='Train loss')
lns4 = ax2.plot(val_loss, color='green', linestyle='-', label='Validation loss')
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='right')
# ax.legend(loc=0)
ax.grid()
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.05)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(0, 9)
ax.set_ylim(0.50, 1)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
ax2.set_ylabel("Loss")
ax2.set_ylim(0, 1.2)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig('ResNet50_result.png')
plt.tight_layout()

print('savefig...')
pdf.savefig()
plt.close()
pdf.close()

with open("ResNet50.txt", 'a+') as f:
    f.write('acc\n')
    for item in acc:
        f.write("{}\n".format(item))
    f.write('val_acc\n')
    for item in val_acc:
        f.write("{}\n".format(item))
    f.write('loss\n')
    for item in loss:
        f.write("{}\n".format(item))
    f.write('val_loss\n')
    for item in val_loss:
        f.write("{}\n".format(item))


def read_image(file_path):
    from PIL import Image
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img.resize((ROWS, COLS), Image.NEAREST)

def predict():
    result = []
    model = keras.models.load_model('ResNet50.h5')
    test_images = [test_dir + '/' + str(i) + '.jpg' for i in range(1, 12501)]
    count = len(test_images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(test_images):
        image = read_image(image_file)
        data[i] = np.asarray(image) / 255.0
        if i % 250 == 0: print('处理 {} of {}'.format(i, count))

    test = data
    predictions = model.predict(test, verbose=1)
    print(predictions)
    for i in range(len(predictions)):
        dog_pre = predictions[i, 1]
        if dog_pre <= 0.005:
            result.append(0.005)
        elif dog_pre >=0.995:
            result.append(0.995)
        else:
            result.append(dog_pre)
        # if predictions[i, 0] >= 0.5:
        #     result.append(0.005)
        # else:
        #     result.append(0.995)
    return result

result = predict()
print(result)

import pandas as pd
# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'id': [i for i in range(1, 12501)], 'label': result})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("ResNet50_result.csv", index=False, sep=',')

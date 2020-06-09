import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import matplotlib.pyplot as plt
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dir = './train2'
validation_dir = './validation'
test_dir = './test1'
ROWS = 299
COLS = 299
CHANNELS = 3
batch_size = 32
epochs = 10

# 构建网络
model = keras.models.Sequential()
# 第一个卷积层，32个卷积核，大小5×5，卷积模式SAME，激活函数relu，输入张量的大小
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层，池化核大小2×2
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(rate=0.25))
# 全连接层，展开操作
model.add(keras.layers.Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(rate=0.25))
# 输出层
model.add(keras.layers.Dense(2, activation='softmax'))
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='CNN_model.pdf')
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

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

model.save('CNN.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc = [
# 0.5309397257394553,
# 0.5508487306624408,
# 0.5741825647203346,
# 0.5836963597230886,
# 0.599769666017215,
# 0.6061288868889693,
# 0.6073807020209105,
# 0.6098342596765309,
# 0.6163436983656103,
# 0.615442391467628]
# val_acc = [
# 0.5835732597494323,
# 0.6047304067756972,
# 0.6193625977269201,
# 0.6019242331938172,
# 0.6241731810937737,
# 0.622970535272968,
# 0.6205652433924121,
# 0.6392062535910072,
# 0.6061334936263734,
# 0.6305872918991269]
# loss = [
# 0.6909083549742432,
# 0.6849430586625445,
# 0.6781208774256877,
# 0.6725511382932292,
# 0.6655056324380465,
# 0.6601159343028975,
# 0.6562990301940278,
# 0.6516339600930843,
# 0.6461074912197677,
# 0.6437906253562299]
# val_loss = [
# 0.6833083502417344,
# 0.6746503720304535,
# 0.664838136413289,
# 0.6610809826951247,
# 0.6495129389355717,
# 0.6452544187706156,
# 0.6393952336954578,
# 0.6323889979238045,
# 0.6498695561058608,
# 0.6289856282736166]

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('CNN_result.pdf')

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
ax.legend(lns, labs, loc='lower right')
# ax.legend(loc=0)
ax.grid()
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.01)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(0, 9)
ax.set_ylim(0.52, 0.64)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.01))
ax2.set_ylabel("Loss")
ax2.set_ylim(0.61, 0.70)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig('CNN_result.png')
plt.tight_layout()
pdf.savefig()
plt.close()
pdf.close()



with open("CNN.txt", 'a+') as f:
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
    model = keras.models.load_model('CNN.h5')
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
dataframe.to_csv("CNN_result.csv", index=False, sep=',')

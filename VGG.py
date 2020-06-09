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
ROWS = 224
COLS = 224
CHANNELS = 3
batch_size = 32
epochs = 10

Inp = keras.layers.Input((ROWS, COLS, CHANNELS))
VGG16_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, CHANNELS))
VGG16_layers = VGG16_model(Inp)
VGG16_layers = keras.layers.GlobalAveragePooling2D()(VGG16_layers)
dense = keras.layers.Dense(128, activation='relu')(VGG16_layers)
output = keras.layers.Dense(2, activation='softmax')(dense)
model = keras.Model(inputs=Inp, outputs=output)
keras.utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file='VGG_model.pdf')
for layer in VGG16_model.layers:
    layer.trainable = False

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 20,        # 随机错切变换
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

model.save('VGG.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc = [
# 0.5260627910485105,
# 0.6336187471849082,
# 0.6765810425146265,
# 0.6964598668098545,
# 0.7143357868924842,
# 0.7238495818967305,
# 0.738771218269471,
# 0.7419758650072408,
# 0.7530419107706174,
# 0.7570477191958142]
# val_acc = [
# 0.5975145318269682,
# 0.6664662257443212,
# 0.6516335937393335,
# 0.7348165965840104,
# 0.732411304787085,
# 0.7568651030598387,
# 0.7450390858219152,
# 0.7733012627183757,
# 0.78492683887864,
# 0.7957506513734168]
# loss = [
# 0.6911738577546119,
# 0.6665668679068034,
# 0.6473838115101387,
# 0.6295454607012991,
# 0.613531503451258,
# 0.5977059371941413,
# 0.5839891065450981,
# 0.5711350576627249,
# 0.5572519428883657,
# 0.5461045492722689]
# val_loss = [
# 0.675842485116751,
# 0.6496889990253949,
# 0.632202279128061,
# 0.6052971238434303,
# 0.5885781505036675,
# 0.5677354072826758,
# 0.5566171285718542,
# 0.5349140995417121,
# 0.5179454926427511,
# 0.5026186188516026]

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('VGG_result.pdf')
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
y_major_locator = MultipleLocator(0.05)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(0, 9)
ax.set_ylim(0.50, 0.80)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.02))
ax2.set_ylabel("Loss")
ax2.set_ylim(0.44, 0.70)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig('VGG_result.png')
plt.tight_layout()

print('savefig...')
pdf.savefig()
plt.close()
pdf.close()


with open("VGG.txt", 'a+') as f:
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
    model = keras.models.load_model('VGG.h5')
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
dataframe.to_csv("VGG_result.csv", index=False, sep=',')
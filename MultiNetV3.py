import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import matplotlib.pyplot as plt
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

ROWS = 299
COLS = 299
CHANNELS = 3
batch_size = 32
epochs = 10

train_dir = './train2'
validation_dir = './validation'
test_dir = './test1'

Inp = keras.layers.Input((ROWS, COLS, CHANNELS))
InceptionV3_model = keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, CHANNELS))
Xception_model = keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, CHANNELS))


InceptionV3_layers = InceptionV3_model(Inp)
InceptionV3_layers = keras.layers.GlobalAveragePooling2D()(InceptionV3_layers)
Xception_layers = Xception_model(Inp)
Xception_layers = keras.layers.GlobalAveragePooling2D()(Xception_layers)

x = keras.layers.Concatenate()([InceptionV3_layers, Xception_layers])
x = keras.layers.Dense(128, activation='relu')(x)
output = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs=Inp, outputs=output)


for layer in InceptionV3_model.layers:
    layer.trainable = False
for layer in Xception_model.layers:
    layer.trainable = False
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='MultiNetV3_model.pdf')

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


## Callback for loss logging per epoch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

lossHistory = LossHistory()


history = model.fit_generator(
    generator = train_generator,
    steps_per_epoch=len(train_generator),
    epochs = epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks = [lossHistory, early_stopping])


model.save('MultiNetV3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc = [
# 0.9081167693184921,
# 0.9579890841720494,
# 0.9595413349356567,
# 0.9601422061989885,
# 0.9630464173050924,
# 0.9662510640428621,
# 0.9657002653863003,
# 0.9686044764909119,
# 0.9704070902809073,
# 0.9656001201742527]
# val_acc = [
# 0.9783523752254961,
# 0.9811585488073762,
# 0.9823611945206571,
# 0.9827620765684506,
# 0.982361194628182,
# 0.9851673682100621,
# 0.9837642814191221,
# 0.9853678091801964,
# 0.9847664861622686,
# 0.9865704548934773]
# loss = [
# 0.334773152772656,
# 0.1595503363571231,
# 0.12556578537762525,
# 0.11194259312281228,
# 0.10345857460571276,
# 0.0945335676809931,
# 0.09526535401710326,
# 0.08729277510378962,
# 0.08348125412530055,
# 0.08723236685270404]
# val_loss = [
# 0.1513840030407614,
# 0.09282822789440272,
# 0.07341845128233482,
# 0.06297009068958075,
# 0.05743915125530524,
# 0.05148896684739078,
# 0.051636338488759405,
# 0.046791229066456705,
# 0.04542942004796139,
# 0.04191058876519692]

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('MultiNetV3_result.pdf')
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
# ax.legend(lns, labs, loc=0)
ax.grid()
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.01)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(0, 9)
ax.set_ylim(0.90, 0.99)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.02))
ax2.set_ylabel("Loss")
ax2.set_ylim(0.04, 0.34)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig('MultiNetV3_result.png')
plt.tight_layout()

print('savefig...')
pdf.savefig()
plt.close()
pdf.close()

with open("MultiNetV3.txt", 'a+') as f:
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
    model = keras.models.load_model('MultiNetV3.h5')
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
dataframe.to_csv("MultiNetV3_result.csv", index=False, sep=',')
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
ROWS = 150
COLS = 150
CHANNELS = 3
batch_size = 32
epochs = 10

# 构建网络
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Dense(2, activation='softmax'))
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='CNN_model_2.pdf')
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
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
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs = epochs,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator)
                    )

model.save('CNN_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# acc = [0.6021731510705424,
# 0.7306594562115066,
# 0.7759751639892746,
# 0.806068799762636,
# 0.8318061188753495,
# 0.848880877275029,
# 0.8621501176706224,
# 0.869410645435882,
# 0.8755695758850333,
# 0.882529668018627]
# val_acc = [
# 0.7017438364282211,
# 0.7532571655735268,
# 0.7943475644510575,
# 0.803367408346045,
# 0.8532772099094846,
# 0.8186009219448321,
# 0.7797153737387789,
# 0.8598917619239164,
# 0.8783323310448522,
# 0.8510723590826935]
# loss = [
# 0.6661569641740851,
# 0.5435218642452293,
# 0.47866486677224046,
# 0.4273717050905501,
# 0.3851066147249633,
# 0.35684207259823736,
# 0.33106576886221956,
# 0.31121891328621853,
# 0.3041782280765393,
# 0.29370824344335694]
# val_loss = [
# 0.577000819112285,
# 0.4954960583732515,
# 0.46103021508468417,
# 0.4392431575015686,
# 0.3440705707745123,
# 0.4299271015682209,
# 0.4768224167283778,
# 0.3479323984088485,
# 0.28914047444624186,
# 0.33843300218428274]

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('CNN_2_result.pdf')

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
ax.set_ylim(0.55, 0.90)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.set_ylabel("Loss")
ax2.set_ylim(0.15, 0.70)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig('CNN_2_result.png')
plt.tight_layout()
pdf.savefig()
plt.close()
pdf.close()


with open("CNN_2.txt", 'a+') as f:
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
    model = keras.models.load_model('CNN_2.h5')
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
dataframe.to_csv("CNN_2_result.csv", index=False, sep=',')

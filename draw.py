import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('pic43.pdf')
from PIL import Image
import os
train_path = 'train'
train2_path = 'train2'
validation_path = 'validation'
test1_path = 'test1'

# train_file = [train_path + '/cat/' + img for img in os.listdir(train_path + '/cat')] + \
#              [train_path + '/dog/' + img for img in os.listdir(train_path + '/dog')]
#
# train_width = []
# train_height = []
# for file in train_file:
#     im = Image.open(file)
#     train_width.append(im.size[0])
#     train_height.append(im.size[1])
#
# plt.figure()
# plt.scatter(train_width, train_height)
# plt.xlabel('width')
# plt.ylabel('height')
# plt.title('scatter diagram of picture size in train dataset')
# # plt.show()
#
# plt.tight_layout()
#
# print('savefig...')
# pdf.savefig()
# plt.close()
# pdf.close()


# test1_file = [test1_path + '/' + img for img in os.listdir(test1_path)]
#
# test1_width = []
# test1_height = []
# for file in test1_file:
#     im = Image.open(file)
#     test1_width.append(im.size[0])
#     test1_height.append(im.size[1])
#
# plt.figure()
# plt.scatter(test1_width, test1_height)
# plt.xlabel('width')
# plt.ylabel('height')
# plt.title('scatter diagram of picture size in test dataset')
# # plt.show()
# plt.tight_layout()
#
# print('savefig...')
# pdf.savefig()
# plt.close()
# pdf.close()

import numpy as np

plt.figure(figsize=(15,5))

# 构建数据
model = ['CNN', 'CNN_2', 'VGG16', 'ResNet50', 'InceptionV3', 'Xception', 'MultiNetV1', 'MultiNetV2', 'MultiNetV3', 'MultiNetV4']
acc =      [0.6154, 0.8825, 0.7570, 0.9579, 0.9485, 0.9682, 0.9662, 0.9736, 0.9656, 0.9751]
val_acc =  [0.6305, 0.8511, 0.7957, 0.9697, 0.9774, 0.9878, 0.9852, 0.9852, 0.9866, 0.9862]
loss =     [0.6438, 0.2937, 0.5461, 0.1075, 0.1326, 0.0936, 0.0990, 0.0735, 0.0872, 0.0659]
val_loss = [0.6290, 0.3384, 0.5026, 0.0803, 0.0750, 0.0514, 0.0521, 0.0438, 0.0419, 0.0380]
kaggle =   [0.62579,0.34389,0.50486,0.09618,0.08515,0.06547,0.06565,0.06082,0.05838,0.05468]
bar_width = 0.17
plt.bar(x=np.arange(len(acc)) - 2 * bar_width, height=acc, label='acc', color='indianred', alpha=0.8, width=bar_width)
plt.bar(x=np.arange(len(val_acc)) - bar_width, height=val_acc, label='val_acc', color='orange', alpha=0.8, width=bar_width)
plt.bar(x=model, height=loss, label='loss', color='steelblue', alpha=0.8, width=bar_width)
plt.bar(x=np.arange(len(val_loss)) + bar_width, height=val_loss, label='val_loss', color='gold', alpha=0.8, width=bar_width)
plt.bar(x=np.arange(len(kaggle)) + 2 * bar_width, height=val_loss, label='kaggle', color='green', alpha=0.8, width=bar_width)

# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for x, y in enumerate(acc):
#     plt.text(x - 2 * bar_width, y + 0.02, '%s' % y, ha='center', va='bottom')
# for x, y in enumerate(val_acc):
#     plt.text(x - bar_width, y + 0.01, '%s' % y, ha='center', va='bottom')
# for x, y in enumerate(loss):
#     plt.text(x, y + 0.08, '%s' % y, ha='center', va='bottom')
for x, y in enumerate(val_loss):
    plt.text(x + bar_width, y + 0.06, '%s' % y, ha='center', va='bottom')
for x, y in enumerate(kaggle):
    plt.text(x + 2 * bar_width, y + 0.01, '%s' % y, ha='center', va='bottom')

# 具体数值
for x, y in enumerate(acc):
    plt.text(x - 2 * bar_width, y + 0.02, '%s' % y, ha='center', va='bottom')
for x, y in enumerate(val_acc):
    if x == 0:
        plt.text(x - bar_width, y + 0.04, '%s' % y, ha='center', va='bottom')
    elif x == 1 or x == 2:
        plt.text(x - bar_width, y + 0.01, '%s' % y, ha='center', va='bottom')
    else:
        plt.text(x - bar_width, y + 0.04, '%s' % y, ha='center', va='bottom')
for x, y in enumerate(loss):
    if x == 0:
        plt.text(x, y + 0.08, '%s' % y, ha='center', va='bottom')
    elif x == 1:
        plt.text(x, y + 0.03, '%s' % y, ha='center', va='bottom')
    elif x == 4:
        plt.text(x, y + 0.04, '%s' % y, ha='center', va='bottom')
    elif x == 5:
        plt.text(x, y + 0.06, '%s' % y, ha='center', va='bottom')
    elif x == 6:
        plt.text(x, y + 0.05, '%s' % y, ha='center', va='bottom')
    elif x == 8:
        plt.text(x, y + 0.05, '%s' % y, ha='center', va='bottom')
    else:
        plt.text(x, y + 0.07, '%s' % y, ha='center', va='bottom')


plt.ylim([0, 1.1])
# 设置标题
plt.title("Data comparison of each experiment")
# 为两条坐标轴设置名称
# plt.xlabel("年份")
# plt.ylabel("销量")
# 显示图例
plt.legend()
# plt.show()
plt.tight_layout()

print('savefig...')
pdf.savefig()
plt.close()
pdf.close()
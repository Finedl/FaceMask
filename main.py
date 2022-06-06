# import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os



# 构造参数解析器（默认，但一些图片格式不会被识别，需要替换或者删除图像）
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# 定义超参数
# 初始化初始学习率（INIT_LR）、要训练的EPOCHS数和Batch（BS）
INIT_LR = 1e-4 #Initial Learning Rate. Later, we will be applying a learning rate decay schedule, thats why it's initial.
EPOCHS = 20 
BS = 32



# 获取数据集目录中的图像列表，然后初始化
# 数据列表（即图像）
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
print(imagePaths)
data = []
labels = []

# 循环图像路径
for imagePath in imagePaths:
	# 从文件名中提取类标签
	label = imagePath.split(os.path.sep)[-2] # 拆分with_mask和without_mask
	# 加载图像和预处理
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	# 分别更新数据和标签列表
	data.append(image)
	labels.append(label)
# 将数据和标签转换为 NumPy 数组
data = np.array(data, dtype="float32")
labels = np.array(labels)



# 对标签执行 one-hot 编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# 使用80%的数据进行训练，剩下的20%用于测试
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# 构建用于数据增强的训练图像
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")



# 加载 MobileNetV2 网络
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
	layer.trainable = False


# 编译模型
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# 训练
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS, epochs=EPOCHS)



# 预测
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
# 显示结果
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
# 保存模型
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")


# 常用的画图
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
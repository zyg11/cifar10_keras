#用keras 实现cifar10
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import MaxPooling2D,Conv2D
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#下载数据
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print("x_train shape: ",x_train.shape)
print(x_train.shape[0],"x_trian samples: ")#样本数目，宽，高，通道数
img_rows,img_cols=32,32
number=4000
x_train = x_train[0:number]
y_train = y_train[0:number]
x_test=x_test[0:500]
y_test=y_test[0:500]


# x_train=x_train.reshape(number,img_rows,img_cols,1)
# x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
#对训练和测试数据处理，转为float
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
#对数据进行归一化到0-1 因为图像数据最大是255
x_train=x_train/255
x_test=x_test/255
#一共10类
nb_classes=10
# 将标签进行转换为one-shot
y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)
#搭建网络
model=Sequential()
# 2d卷积核，包括32个3*3的卷积核
# 因为X_train的shape是【样本数，通道数，图宽度，图高度】这样排列的，
# 而input_shape不需要（也不能）指定样本数。
model.add(Conv2D(32,(3,3),border_mode='same',
                 input_shape=x_train.shape[1:],
                 activation='relu'))#指定输入数据的形状
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))#最大池化
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))#最大池化

model.add(Dropout(0.5))#进行Dropout
model.add(Flatten())#压扁平准备全连接
#全连接
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
#输出层
model.add(Dense(nb_classes,activation='softmax'))
model.summary()
sgd=SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True)
model.compile(optimizer=sgd,
              loss="categorical_crossentropy",
              metrics=['accuracy'])
#train
batch_size=32
nb_epoch=50
data_augmentation=False#是否数据扩充，主要针对样本过小方案
#如果够多，不需要数据扩充
if not data_augmentation:
    print('Not use data augmentation')
    result=model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(x_test,y_test),
        shuffle=True
    )
else:
    print('Using real_time data augmentation')
    # this will do preprocessing and realtime data augmentation
    datagen=ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=True,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)
    data_genetator=datagen.flow_from_directory(
        x_train,
        y_train,
        batch_size=batch_size
    )
    result=model.fit(
        data_genetator,
        samples_per_epoch=3000,
        nb_epoch=nb_epoch,
        validation_data=(x_test,y_test)
    )
model.save('E:/keras_data/cifar10/simple_cifar10_model.h5')
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()
#训练的acc_loss图
plot_training(result)

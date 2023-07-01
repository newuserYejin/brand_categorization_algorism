import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = "..\\Brand_photos_data"   # 저장폴더
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#폴더에 있는 그림 출력하기
#roses = list(data_dir.glob('roses/*'))
#print(str(roses[1]))
#picture=PIL.Image.open(str(roses[1]))
#ishape=np.array(picture)
#print(ishape.shape)
#picture.show()

batch_size = 32
img_width = 200
img_height = 180

###############################################################################
# 학습,검증데이터 로드 tf.data.Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(train_ds.element_spec)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):     #몇개의 batch를 가져올지
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
#plt.show()
#plt.imshow(images[0].numpy().astype("uint8"))
#plt.show()
##################################################################################

##성능을 높이도록 데이터세트 구성하기
#AUTOTUNE = tf.data.experimental.AUTOTUNE
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#데이터 표준화하기,0~1사이값으로 변환
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 하나의 batch를 선택해 평균값표시
[(temp_img, temp_label)] = normalized_train_ds.take(1)
first_image = temp_img[0]
print("norm_train_ds=",np.min(first_image),"~",np.average(first_image),"~",np.max(first_image))     #0.0 0.96902645

# 모델만들기 ##############################################################
num_classes = len(class_names)          #5
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
    #data_augmentation,    #추가    
    ##layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  #추가
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])
#모델 컴파일하기
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

####################################################################################
#모델 훈련하기
#0~1사이의 float32
#(n,height,width,3)
print(train_ds.element_spec)

[(temp_img, temp_label)] = normalized_train_ds.take(1)
show_img=temp_img[0].numpy()*255
plt.imshow(show_img.astype("uint8"))
plt.show()

epochs=20

model_path = '../model/' + 'Brand-model-' + '-{epoch:04d}-{val_loss:.4f}.hdf5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)
#cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
  normalized_train_ds,
  validation_data=normalized_val_ds,
  epochs=epochs,
  batch_size=50, verbose=1, callbacks=[cb_checkpoint]
)

model.save("../model/"+"brand-model.hdf5")

#모델만 저장=>json
model_json = model.to_json()
with open("../model/"+"model_only.json", "w") as json_file :
	json_file.write(model_json)

#weights만 저장=>HDf5
model.save_weights("../model/"+"weights_only.hdf5")
##weight 출력 test
print('model weight :', model.get_weights())

#훈련 결과 시각화하기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
############################################################################

#test 성능평가
score = model.evaluate(val_ds, verbose=0) # test 값 결과 확인
print('Test loss:', score[0])
print('Test accuracy:', score[1])


##훈련데이터중에 하나를 테스트해보기
## pull one batch from the BatchDataSet
#normalized_train_ds.range(1)
#[(temp_img, temp_label)] = normalized_train_ds.take(1)
#show_img=temp_img[0].numpy()*255
#plt.imshow(show_img.astype("uint8"))
#plt.show()
#temp_norm=tf.expand_dims(temp_img[0], 0)
#predictions = model.predict(temp_norm)
#score = tf.nn.softmax(predictions[0])   # %로 변환
#for i in range(len(class_names)):
#    print('%20s[%d]:%.4f(%.4f%%)'%(class_names[i],i,predictions[0][i],score[i]*100))
#print("max arg=",np.argmax(predictions),"[",np.max(score),"]")
#print(
#    "This image most likely belongs to {} with a {:.2f} percent confidence."
#    .format(class_names[np.argmax(score)], 100 * np.max(score))
#)

#화일 로드 & 변환(keras)
test_image_folder_path = "../test_brand_photo/" 
img = keras.preprocessing.image.load_img(
    test_image_folder_path+'a.jpg', target_size=(img_height, img_width)
)   #interpolation="nearest"(default)
img.show()

img_array = keras.preprocessing.image.img_to_array(img)
norm_img_array = np.array(img_array)/255.0                     # Normalize channel between 0 to 1
norm_img_array = tf.expand_dims(norm_img_array, 0) # Create a batch

predictions = model.predict(norm_img_array)

score = tf.nn.softmax(predictions[0])   # %로 변환
for i in range(len(class_names)):
    print('%20s[%d]:%.4f(%.4f%%)'%(class_names[i],i,predictions[0][i],score[i]*100))
print("max arg=",np.argmax(predictions),"[",np.max(score),"]")
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

import glob
import cv2

test_image_folder_path = "..//test_brand_photo/"
for item in glob.glob(test_image_folder_path+"*"):
    mat_img = cv2.imread(item)
    mat_img=cv2.cvtColor(mat_img,cv2.COLOR_BGR2RGB)
    mat_resizedimg = cv2.resize(mat_img,dsize=(img_width,img_height),interpolation=cv2.INTER_NEAREST)        
    cv2.imshow("Display Window", mat_resizedimg)
    cv2.waitKey(0)

    #mat_resizedimg=np.expand_dims(mat_resizedimg, axis = -1)   #흑백일때만 확장(h,w)=>(h,w,1)
    norm_img_array = mat_resizedimg.astype('float32')
    norm_img_array /= 255.0        #model 안에서 에서 scale
    norm_img_array= np.array([norm_img_array])

    print("norm_img_arra  y shape=>",norm_img_array.shape)                                         #(1, 200, 300, 3)
    print("norm_img_array=>",np.min(norm_img_array),"~",np.average(norm_img_array),"~",np.max(norm_img_array))
    predictions = model.predict(norm_img_array)

    score = tf.nn.softmax(predictions[0])   # %로 변환
    for i in range(len(class_names)):
        print('%20s[%d]:%.4f(%.4f%%)'%(class_names[i],i,predictions[0][i],score[i]*100))

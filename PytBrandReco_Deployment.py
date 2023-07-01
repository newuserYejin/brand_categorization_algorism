import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2

img_width = 200
img_height = 180
class_names=['adidas[0]', 'fila[1]', 'lecoq[2]', 'nike[3]','puma[4]']

# 모델 로드
MODEL_SAVE_FOLDER_PATH = "../Model/"
model=tf.keras.models.load_model(MODEL_SAVE_FOLDER_PATH+"brand-model.hdf5")
model.summary()

##모델만 로드 <= json
## json으로 만든 model은 따로 함수를 또 import를 해야함
##from keras.models import model_from_json
#json_file = open("model_only.json", "r") 
#json_model = tf.keras.json_file.read() 
#json_file.close() 
#model = model_from_json(json_model)
#
##weights만 로드 <= HDf5
#model.load_weights('weights_only.h5')
##weight 출력 test
#print('model weight :', model.get_weights())


model_json = model.to_json()
with open("../model/"+"tmodel_only.json", "w") as json_file :
	json_file.write(model_json)
#weights만 저장=>HDf5
model.save_weights("../model/"+"tweights_only.hdf5")




#sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
#sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
#img = keras.preprocessing.image.load_img(
#    sunflower_path, target_size=(img_height, img_width)
#)

#화일 로드 & 변환
test_image_folder_path = "../test_brand_photo/"
img = keras.preprocessing.image.load_img(
    test_image_folder_path+'a.jpg', target_size=(img_height, img_width)
)   #interpolation="nearest"(default)
img.show()

img_array = keras.preprocessing.image.img_to_array(img)
norm_img_array = np.array(img_array)/255.0
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

test_image_folder_path = "../test_brand_photo/"
for item in glob.glob(test_image_folder_path+"*"):
    mat_img = cv2.imread(item)
    mat_img=cv2.cvtColor(mat_img,cv2.COLOR_BGR2RGB)
    mat_resizedimg = cv2.resize(mat_img,dsize=(img_width,img_height),interpolation=cv2.INTER_NEAREST)        
    cv2.imshow("Display Window", mat_resizedimg)
    cv2.waitKey(0)
    
    #mat_resizedimg=np.expand_dims(mat_resizedimg, axis = -1)   #흑백일때만 확장(h,w)=>(h,w,1)
    norm_img_array = mat_resizedimg.astype('float32')
    norm_img_array /= 255.0                                    #model 안에서 에서 scale
    norm_img_array= np.array([norm_img_array])


    print(norm_img_array[0][0][0][0],norm_img_array[0][0][0][1],norm_img_array[0][0][0][2])
    print(norm_img_array[0][0][1][0],norm_img_array[0][0][1][1],norm_img_array[0][0][1][2])
    print("norm_img_arra  y shape=>",norm_img_array.shape)                                         #(1, 200, 300, 3)
    print("norm_img_array=>",np.min(norm_img_array),"~",np.average(norm_img_array),"~",np.max(norm_img_array))
    predictions = model.predict(norm_img_array)

    score = tf.nn.softmax(predictions[0])   # %로 변환
    for i in range(len(class_names)):
        print('%20s[%d]:%.4f(%.4f%%)'%(class_names[i],i,predictions[0][i],score[i]*100))

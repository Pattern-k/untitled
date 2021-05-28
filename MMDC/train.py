
import tensorflow as tf
import numpy as np
import glob
from model import model



def data_process():

    image = glob.glob("F:/data_segmentation/image/*")

    image.sort(key=lambda x: x.split("/")[-1])
    label = glob.glob("F:/data_segmentation/label/*")

    label.sort(key=lambda x: x.split("/")[-1])


    np.random.seed(2020)
    index = np.random.permutation(len(image))

    image = np.array(image)[index]
    label = np.array(label)[index]


    dataset = tf.data.Dataset.from_tensor_slices((image,label))


    test_count= int(len(image)*0.2)
    train_count= len(image)-test_count

    train_ds = dataset.skip(test_count)
    test_ds = dataset.take(test_count)


    train_ds = train_ds.map(load_img,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_img)

    batch_size = 8

    buffer_size = 200
    STEPS_PER_EPOCH = train_count//batch_size
    VALIDATION_STEPS = test_count//batch_size


    train_ds = train_ds.cache().shuffle(buffer_size).batch(batch_size).repeat()
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    return train_ds,test_ds,STEPS_PER_EPOCH,VALIDATION_STEPS

def read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=3)
    return img
def read_label(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=1)
    return img

def normalize(img,mask):
    img = tf.cast(img,tf.float32)/127.5-1
    mask = tf.cast(mask,tf.float32)
    return img,mask

def load_img(img_path,label_path):
    img = read_img(img_path)
    img = tf.image.resize(img,(256,256))

    label = read_label(label_path)
    label = tf.image.resize(label,(256,256))

    img,label = normalize(img,label)
    return img,label

model = model.create_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.999),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

train_dataset,test_dataset,STEPS_PER_EPOCH,VALIDATION_STEPS =data_process()

epoch =120



history = model.fit(train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs= epoch,
                    validation_data=test_dataset,
                    validation_steps=VALIDATION_STEPS)

model.save_weights("F:/data_segmentation/save_weight/MMDC-Unet.h5")




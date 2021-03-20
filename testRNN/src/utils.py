import numpy as np
from tensorflow.keras import backend as K
import cv2
import os
import shutil
import skimage
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from keract import get_activations



# def gaussian_noise(image, seed, mean=0, var=0.001):
#     '''
#         add gaussian noise to image
#     '''
#     np.random.seed(seed)
#     noise = np.random.normal(mean, var ** 0.5, image.shape)
#     out = image + noise
#     out = np.clip(out, 0.0, 1.0)
#     return out

def hard_sigmoid(x):
    return np.maximum(0, np.minimum(1, 0.2*x+0.5))

def lp_norm(p,n1,n2):
    n1 = np.array([n1]).ravel()
    n2 = np.array([n2]).ravel()
    m = np.count_nonzero(n1-n2)
    return np.linalg.norm(n1-n2,ord=p)/float(m)
    
def l2_norm(n1,n2):
    n1 = np.array([n1]).ravel()
    n2 = np.array([n2]).ravel()
    m = np.count_nonzero(n1-n2)
    return np.linalg.norm(n1-n2,ord=2)/float(m)
    
def getActivationValue(model,layer,test):
    #print("xxxx %s"%(str(self.model.layers[1].input.shape)))
    OutFunc = K.function([model.input], [model.layers[layer].output])
    out_val = OutFunc([test, 1.])[0]
    return np.TCueeze(out_val)
    
def layerName(model,layer):
    layerNames = [layer.name for layer in model.layers]
    return layerNames[layer]
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while suSCess:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', suSCess)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features
    
def extract_vgg16_features_live(model, video_input_file_path):
    # print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    features = []
    images = []
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', suSCess)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            images.append(img)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    images = np.array(images)
    return images, unscaled_features

def add_noise(images, mode, seeds, variance):
    if mode is not None:
        images = skimage.util.random_noise(images / 255.0, mode=mode, var=variance, seed=seeds)
        images = images * 255.0
        images = images.astype(np.uint8)
    return images

def extract_features(model,images):

    input = [img_to_array(image) for image in images]
    input = preprocess_input(np.array(input))
    features = model.predict(input)
    return features



def add_noise_extract_vgg16_features_live(model, images_set, mode, seeds, variance):

    images_set = [add_noise(images, mode, seeds, variance) for images in images_set]
    unscaled_features  = [extract_features(model, images) for images in images_set]

    return images_set, unscaled_features

def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'

    MAX_NB_CLASSES = 11

    input_data_dir_path = os.path.join (data_dir_path, data_set_name)
    output_feature_data_dir_path = os.path.join (data_dir_path, output_dir_path)

    if model is None:
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = os.path.join (input_data_dir_path, f)
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = os.path.join (output_feature_data_dir_path, output_dir_name)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = os.path.join (file_path, ff)
                output_feature_file_path = os.path.join (output_dir_path, ff.split('.')[0] + '.npy')
                x = extract_vgg16_features(model, video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

def Z_ScoreNormalization(x,mu,sigma):
	x = (x - mu) / sigma
	return x

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def delete_folder(path):
    folder = os.path.exists(path)
    if folder:
        shutil.rmtree(path)

def setup_dir_for_file (f):
    dir = os.path.dirname (f)
    if not os.path.exists (dir):
        os.makedirs (dir)
    elif not os.path.isdir (dir):
        raise NotADirectoryError (dir)

def aggregate_inf(h_train, indices):

    alpha1 = np.sum(np.where(h_train > 0, h_train, 0), axis=2)
    alpha2 = np.sum(np.where(h_train < 0, h_train, 0), axis=2)

    alpha11 = np.insert(np.delete(alpha1, -1, axis=1), 0, 0, axis=1)
    alpha22 = np.insert(np.delete(alpha2, -1, axis=1), 0, 0, axis=1)

    alpha_TC = np.abs(alpha1 + alpha2)
    alpha_SC = np.abs(alpha1 - alpha11 + alpha2 - alpha22)

    mean_TC = np.mean(alpha_TC[:,indices])
    std_TC = np.std(alpha_TC[:,indices])

    max_BC = np.max(alpha_TC[:,indices])
    min_BC = np.min(alpha_TC[:,indices])

    max_SC = np.max(alpha_SC[:,indices])
    min_SC = np.min(alpha_SC[:,indices])

    return mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC

def oracle(test1, test2, lp, oracleRadius):
    diff_n = np.count_nonzero(test1 - test2, axis= (1,2))
    diff = np.linalg.norm(test1 - test2, ord=lp,axis=(1,2))
    return diff/diff_n <= oracleRadius, diff

def oracle_uvlc(test1, test2, lp, oracleRadius):

    diff_n = np.array([np.count_nonzero(test1[i] - test2[i]) for i in range(len(test1))])
    diff = np.array([np.linalg.norm(test1[i].flatten('F') - test2[i].flatten('F'), ord=lp) for i in range(len(test1))])
    return diff / diff_n <= oracleRadius, diff / diff_n

def neuron_boudary_judge(feature,ub,lb):
    if feature > ub:
        return 0
    elif feature < lb:
        return 1
    else:
        return 3

def get_activations_single_layer(model,x,layerName):
    act = get_activations(model,x,layer_names = layerName)
    return act[layerName]


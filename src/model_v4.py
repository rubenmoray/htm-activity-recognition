## IMPORT SECTION
import json
import os
import glob
import numpy as np
import argparse
import random
import pickle
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# KERAS
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import img_to_array as keras_img_to_array
from keras import Input, Model, regularizers, Sequential
#from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape, Dense, BatchNormalization, ZeroPadding2D
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Flatten, Dropout, Reshape, Dense, BatchNormalization, ZeroPadding3D, InputLayer
from numba import vectorize, float32, float64
from sklearn.manifold import TSNE
import htm.bindings.encoders
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR, Metrics


## END IMPORT SECTION

## CONSTANTS SECTION
ScalarEncoder = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters
## END CONSTANTS SECTION


## FUNCTIONS SECTION
def auto_encs(shape):
    input_img = Input(shape=(shape[0], shape[1], shape[2], 1))  # adapt this if using `channels_first` image data format
    x = ZeroPadding3D(padding=2)(input_img)
    x = Conv3D(int(16), (3, 3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding3D((1, 1, 1))(x)

    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = Conv3D(int(32), (3, 3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding3D((1, 1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = Conv3D(int(64), (3, 3, 3), activation='selu', padding='same')(x)
    x = ZeroPadding3D((1, 1, 1))(x)
    x = BatchNormalization(axis=-1)(x)


    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
    encoded = Flatten()(encoded)


    encoded = Dense((256), activation='selu', kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dense((128), activation='selu')(encoded)  # Cut of the network here after training.

    encoded = Reshape((4, 4, 8, 1))(encoded)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv3D(int(shape[0]/2), (3, 3, 3), activation='selu', padding='same')(encoded)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(int(shape[0]/4), (3, 3, 3), activation='selu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(int(shape[0]/16), (3, 3, 3), activation='selu',padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling3D((2, 2, 2))(x)

    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    decoded = Reshape((decoded.shape[1],decoded.shape[2],decoded.shape[3]))(decoded)
    decoded = Dense(3,activation='selu')(decoded)
    decoded = Reshape((decoded.shape[1],decoded.shape[2],decoded.shape[3],1))(decoded)
    decoded = UpSampling3D((2, 2, 1))(decoded)
    decoded = UpSampling3D((2, 2, 1))(decoded)

    cnn_embeddings = Model(input_img, decoded)
    cnn_embeddings.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # try as well mse
    print(cnn_embeddings.summary())
    return cnn_embeddings

def save_model(model):
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def get_available_images(category_name, mode, config_file_path, data_directory):
    with open(config_file_path, 'r') as jsonFile:
        annotation_file = json.load(jsonFile)

        ## SUBDIRECTORIES
        # Get list of subdirectories
        subdirectories = [x[0] for x in os.walk(data_directory)]
        

        # Remove first item. It contains the root directory
        subdirectories.pop(0)
        

        # Remove data_directory name from subdirectories
        subdirectories = [x.split(os.path.sep)[-1] for x in subdirectories]
        # Get all images of directory
        aux = {}
        for subdir in subdirectories:
            aux[subdir] = glob.glob('{}/*'.format(os.path.join(data_directory, subdir)))

        subdirectories = aux
        
        # END SUBDIRECTORIES

        ## ANNOTATION FILE
        # Loop annotations file training set
        image_paths = {}
        labels = {}

        # Loop videos
        videos = annotation_file[category_name][mode]['videos']
        for key_videos in videos.keys():
            
            # Check if video folder is in data_directory
            if key_videos in subdirectories:
                subdir_video_frames = subdirectories[key_videos]
                image_paths[key_videos] = []
                labels[key_videos] = []
                if mode == 'train_split':
                    print(key_videos)

                    # Loop frames
                    frames = videos[key_videos]['frames']
                    for key_frame in frames.keys():

                        if os.path.join(data_directory, key_videos, key_frame) in subdir_video_frames:
                            # Append image path and label
                            image_paths[key_videos].append(os.path.join(data_directory, key_videos, key_frame))
                            labels[key_videos].append(frames[key_frame])
                else:
                    print(key_videos)
                    for frame in subdir_video_frames:
                        image_paths[key_videos].append(frame)
                        labels[key_videos].append(-1)

        ## END ANNOTATION FILE

        return image_paths, labels

def load_data(images, image_dims, frames_second, max_frames_second, mode='train_split'):

    result_dataset = {}
    final_labels_result = {}
    #print(images.keys())
    for i in images['frames']:
        image_paths = images['frames'][i]
        labels = images['labels'][i]
        ## LOAD IMAGES

        data = {}
        final_labels = {}
        frames_loaded_number = 0
        for index, img_path in enumerate(image_paths):
                
            # Load maximum number of frames
            if frames_loaded_number < max_frames_second :

                # Random choice True or False. We need to check if this has an impact in the final model
                load_frame = bool(random.getrandbits(1)) 

                # Load frame
                if load_frame == True:
                    frames_loaded_number += 1
                    print('Loading image {}'.format(img_path))
                    img = keras_image.load_img(img_path, target_size=(image_dims[1], image_dims[0]))     
                    img = keras_img_to_array(img)

                    data[os.path.basename(img_path)] = img
                    print('Image {} loaded'.format(img_path))
                    final_labels[os.path.basename(img_path)] = labels[index] # Append selected frame label
            if index % frames_second == 0:
                frames_loaded_number = 0

        result_dataset[i] = data
        final_labels_result[i] = final_labels
        


    return result_dataset, final_labels_result

def pre_data (training_set, test_set, shape):

    print(len(training_set), type(training_set), np.array(training_set).shape)
    # Scale the raw pixel intensities to the range [0, 1]
    x_train = np.array(training_set).astype('float32') / (255 - 1)
    x_test = np.array(test_set).astype('float32') / (255 - 1)
    x_train = np.reshape(x_train, (len(x_train), shape[0], shape[1], shape[2], 1)) 
    x_test = np.reshape(x_test, (len(x_test), shape[0], shape[1], shape[2], 1)) 
    print("[INFO] train matrix: {} images ({:.2f}MB)".format(len(x_train), x_train.nbytes / (1024 * 1000.0)))
    print("[INFO] test matrix: {} images ({:.2f}MB)".format(len(x_test), x_test.nbytes / (1024 * 1000.0)))
    print(x_train.shape, x_test.shape)
    # TODO: Create autoenc
    shape = shape +  (1,)
    print(shape)
    model = auto_encs(shape)
    #model = auto_encs2(shape, shape[0])
    model.fit(x_train, x_train,
              epochs=3,
              batch_size=64,
              shuffle=True,
              validation_data=(x_test, x_test))

    model_new = Model(model.layers[0].input,model.layers[15].output)
    return model_new


# Save model to disk
def save_model(model):
    model_json = model.to_json()
    with open('model/embedding_v3.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/embedding_v3.h5")
    print("Saved model to disk")


# load model from disk
def load_model():
    json_file = open('model/embedding_v3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/embedding_v3.h5")
    print("Loaded model from disk")
    return loaded_model

# Yet to be used K-winner
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


SCALE_FACTOR = 2
MATCH_FACTOR = 138

# Scale vector to be used in conjunction with SDR resolution parameter.
@vectorize(['float32(float32)', 'float64(float64)'], target='parallel')
def scale_vector(x):
    return x * SCALE_FACTOR


# Get the output from auto encoder and prep for SDR generation
def predict_and_reduce_v2(train_dataset, test_dataset,new_model=None):
    pred = []
    norms = []
    if new_model is None:
        return 0

    train_dataset = np.array(train_dataset, dtype=np.float32)
    test_dataset = np.array(test_dataset, dtype=np.float32)

    
    train_dataset = np.reshape(train_dataset, (len(train_dataset), shape[0], shape[1], shape[2], 1))
    test_dataset = np.reshape(test_dataset, (len(test_dataset), shape[0], shape[1], shape[2], 1))

    #concatenate train and test to feed into HTM
    concat_dataset = np.concatenate([train_dataset,test_dataset],axis=0)

    predictions = new_model.predict(concat_dataset)

    print(predictions.shape)
    for vec in predictions:
        norms.append(np.linalg.norm(vec))
    # Used TSNE to reduce dimensions for data prep for  RDSE encoder
    tsne = TSNE(n_components=2)

    X_hat = tsne.fit_transform(predictions)

    X_hat = scale_vector(X_hat)
    # Get the norm of the vector (128) as input to RDSE
    # Adding this parameter helps SP get better results
    norms = np.array(norms).reshape(-1, 1)
    # Serialise the output for the next stage.
    X_hat = np.concatenate([X_hat, norms], axis=1)
    pickle.dump(X_hat, open('data/x_hat_v3.pkl', mode='wb'))





# Setup the RDSE encoder for imput SDR
# In this method we use diffreent RDSE for encoding all vector as SDR
def scaler_data_randonscaler_method_2():
    pooler_data = []
    data = pickle.load(open('data/x_hat_v3.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    col4 = data[:, 3:4].flatten()
    parameter1 = RDSE_Parameters()
    parameter1.size = 2000
    parameter1.sparsity = 0.02
    parameter1.resolution = 0.66
    rsc1 = RDSE(parameter1)
    parameter2 = RDSE_Parameters()
    parameter2.size = 2000
    parameter2.sparsity = 0.02
    parameter2.resolution = 0.66
    rsc2 = RDSE(parameter2)
    parameter3 = RDSE_Parameters()
    parameter3.size = 2000
    parameter3.sparsity = 0.02
    parameter3.resolution = 0.66
    rsc3 = RDSE(parameter3)
    parameter4 = RDSE_Parameters()
    parameter4.size = 2000
    parameter4.sparsity = 0.02
    parameter4.resolution = 0.66
    rsc4 = RDSE(parameter4)
    # Create SDR for 3D TSNE input plus one for magnitude for 128 D original vector.
    for _x1, _x2, _x3, _x4 in zip(col1, col2, col3, col4):
        x_x1 = rsc1.encode(_x1)
        x_x2 = rsc2.encode(_x2)
        x_x3 = rsc3.encode(_x3)
        x_x4 = rsc4.encode(_x4)
        pooler_data.append(SDR(8000).concatenate([x_x1, x_x2, x_x3, x_x4]))
    return pooler_data


# Create SP
def spatial_pooler_encoder(pooler_data):
    sp1 = SpatialPooler(
        inputDimensions=(8000,),
        columnDimensions=(8000,),
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=0.0335,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13999999999999999,
        boostStrength=4.0,
        wrapAround=True
    )
    sdr_array = []
    # We run SP over there epochs and in the third epoch collect the results
    # this technique yield betters results than  a single epoch
    for encoding in pooler_data:
        activeColumns1 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns1)
    for encoding in pooler_data:
        activeColumns2 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns2)
    for encoding in pooler_data:
        activeColumns3 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns3)
        sdr_array.append(activeColumns3)


    train_dataset = np.array(train_dataset, dtype=np.float32)
    test_dataset = np.array(test_dataset, dtype=np.float32)
    
    train_dataset = np.reshape(train_dataset, (len(train_dataset), shape[0], shape[1], shape[2], 1))
    test_dataset = np.reshape(test_dataset, (len(test_dataset), shape[0], shape[1], shape[2], 1))


    counter = 0
    # finally we loop over the SP SDR and related image to get the once
    # which have a greater overlap with the image we are searching for
    for _s, _img in zip(sdr_array, test_dataset):
        _x_ = hold_out2.getOverlap(_s)
        if _x_ > MATCH_FACTOR : # Adjust as required.
            _img_ = _img.reshape((shape[0], shape[1]))
            _img_ = (_img_ * 254).astype(np.uint8)
            im = Image.fromarray(_img_).convert('RGB')
            im.save('test_results/' + str(counter) + 'outfile.jpg')
            print('Sparsity - ' + str(_s.getSparsity()))
            print(_x_)
            print(str('counter - ') + str(counter))
            counter += 1
            # Write all images to file which have good overlap with the target  image


def spatial_pooler_encoder_v2(train_dataset,test_dataset,train_labels,test_labels):

    data = pickle.load(open('data/x_hat_v3.pkl', mode='rb'))
    col1 = data[:, 0:1].flatten()
    col2 = data[:, 1:2].flatten()
    col3 = data[:, 2:3].flatten()
    parameter1 = RDSE_Parameters()
    parameter1.size = 2000
    parameter1.sparsity = 0.02
    parameter1.resolution = 1 
    rsc1 = RDSE(parameter1)
    parameter2 = RDSE_Parameters()
    parameter2.size = 2000
    parameter2.sparsity = 0.02
    parameter2.resolution = 1
    rsc2 = RDSE(parameter2)
    parameter3 = RDSE_Parameters()
    parameter3.size = 2000
    parameter3.sparsity = 0.02
    parameter3.resolution = 1
    rsc3 = RDSE(parameter3)
    pooler_data = []
    for _x1, _x2, _x3 in zip(col1, col2, col3):
        x_x1 = rsc1.encode(_x1)
        x_x2 = rsc2.encode(_x2)
        x_x3 = rsc3.encode(_x3)
        pooler_data.append(SDR(6000).concatenate([x_x1, x_x2, x_x3]))

    sp1 = SpatialPooler(
        inputDimensions=(6000,),
        columnDimensions=(6000,),
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=0.0335,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13999999999999999,
        boostStrength=4.0,
        wrapAround=True
    )

    sdr_array = []
    for encoding in pooler_data:
        activeColumns1 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns1)
    for encoding in pooler_data:
        activeColumns2 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns2)
    for encoding in pooler_data:
        activeColumns3 = SDR(sp1.getColumnDimensions())
        sp1.compute(encoding, True, activeColumns3)
        sdr_array.append(activeColumns3)

    hold_out = sdr_array[-1]

    train_dataset = np.array(train_dataset, dtype=np.float32)
    test_dataset = np.array(test_dataset, dtype=np.float32)


    testing_labels = []
    training_labels=[]
    overlap_score=[]
    for test_index in range(len(test_dataset)):
        test_sdr_array = sdr_array[len(train_dataset)+test_index]
        train_index=0
        for sdr, actual_img in zip(sdr_array, np.array(train_dataset)):
            overlap = test_sdr_array.getOverlap(sdr)
            sparsity = test_sdr_array.getSparsity()
            testing_labels.append(test_labels[test_index])
            training_labels.append(train_labels[train_index])
            overlap_score.append(overlap)

            train_index+=1

    #make dataframe to contain results and save as CSV
    df = pd.DataFrame({'test_label':testing_labels,'overlap_score':overlap_score,'train_label':training_labels})
    df.to_csv('results.csv')

    return print('Made CSV file!')
        
## END FUNCTIONS SECTION
if __name__ == '__main__':
    description = 'Script to load the images from the Annotation file into an autoencoder.'
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("-n", "--video_names", help="Name of the videos names in the annotation file (contiguous_videos, long_gap, short_gap)")
    parser.add_argument("-a", "--annotation_file_path", help="Annotation file path")
    parser.add_argument("-p", "--video_path", help="Path to the video frames.")
    parser.add_argument("-w", "--image_width", help="Image width in pixels.")
    parser.add_argument("-e", "--image_height", help="Height of the image in pixels.")
    parser.add_argument("-d", "--image_dimensions", help="Dimesions of the image (Example: 3 (rgb)).")
    parser.add_argument("-s", "--frames_second", help="Frames per second of the videos.")
    parser.add_argument("-f", "--max_frames_second", help="Maximum number of frames to load by the training model.")


    
    args = parser.parse_args()

    # COMMENT OUT WHEN U NEED TO RUN 
    #args = argparse.Namespace(annotation_file_path='./Annotation.json', frames_second='30', image_dimensions='3', image_height='128', image_width='128', max_frames_second='5', video_names='contiguous_videos,short_gap', video_path='./videos')

    
    image_dims = (int(args.image_width), int(args.image_height), int(args.image_dimensions))
    shape = image_dims
    #model = auto_encs(shape)

    # CHANGES DIRECTORY.... UNCOMMENT WHEN RUNNING 
    #os.chdir('C:\data-dissertation\code\IJCAI-2021-Continual-Activity-Recognition-Challenge-main\htm-video-analysis\src')


    images = {}
    video_names = args.video_names
    aux = video_names.split(",")
    if len(aux) != 2:
        raise Exception("Need two video names")

    for a in aux:
        video_path = os.path.join(args.video_path, a, 'frames')
        if os.path.isdir(video_path):
            image_paths, labels = get_available_images(a, 'train_split', args.annotation_file_path, video_path) # We load only train for testing
            images[a] = {"frames": image_paths, "labels": labels}

    # This is for testing. Replace list index by the name of the video. For this demo we use the first two videos and only the train split.
    datasets_dict = {}
    labels_dict = {}
    for d in images:
        dataset, l =  load_data(images[d], image_dims, int(args.frames_second), int(args.max_frames_second))

        datasets_dict.update(dataset)
        labels_dict.update(l)
    # Split train and labels
    ## Load images and labels in 1 list
    datasets = []
    labels = []
    
    for d in datasets_dict:
        for f in datasets_dict[d]:
            datasets.append(datasets_dict[d][f])
            labels.append(labels_dict[d][f])


    #normalize dataset
    scaler = MinMaxScaler()

    # rewrite as array
    datasets = np.array(datasets)

    datasets_2d = np.reshape(datasets, (len(datasets)*128,128*3))
    dataset_2d_scaled = scaler.fit_transform(datasets_2d)
    datasets_scaled = np.reshape(dataset_2d_scaled,[len(datasets),128,128,3])

    # split dataset
    train_dataset, test_dataset, train_labels, test_labels = train_test_split(datasets_scaled, labels, test_size=0.5, random_state=42)




    print(len(train_dataset), len(test_dataset))
    ## END Split

    model = pre_data(train_dataset, test_dataset, shape)

    predict_and_reduce_v2(train_dataset, test_dataset ,model)

    spatial_pooler_encoder_v2(train_dataset,test_dataset,train_labels,test_labels)

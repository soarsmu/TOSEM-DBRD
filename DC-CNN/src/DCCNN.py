import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
from datetime import datetime
from utils import get_logger
import argparse
from modules import represent_training_pairs

SEED = 42


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seeds(seed=42):
    """
    This function is to set seeds for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

def set_global_determinism(seed=42):
    """
    This function is to set seeds for reproducibility
    """
    set_seeds(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # new flag present in tf 2.0+

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(SEED)

def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def DCCNN_Model(input_shape):
    X_input = Input(shape=input_shape)
    
    X_1 = Conv2D(100, kernel_size=(1,20), strides=(1,1), activation='relu')(X_input)
    X_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_1)
    X_1 = Reshape((300,100,1))(X_1)
    
    X_1_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1), activation='relu')(X_1)
    X_1_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_1)
    X_1_1 = MaxPooling2D(pool_size=(300,1), padding='valid')(X_1_1)
    X_1_1 = Flatten()(X_1_1)
    X_1_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1), activation='relu')(X_1)
    X_1_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_2)
    X_1_2 = MaxPooling2D(pool_size=(299,1), padding='valid')(X_1_2)
    X_1_2 = Flatten()(X_1_2)
    X_1_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1), activation='relu')(X_1)
    X_1_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_3)
    X_1_3 = MaxPooling2D(pool_size=(298,1), padding='valid')(X_1_3)
    X_1_3 = Flatten()(X_1_3)
    
    X_1 = tf.keras.layers.Concatenate(axis=-1)([X_1_1,X_1_2])
    X_1 = tf.keras.layers.Concatenate(axis=-1)([X_1,X_1_3])
    
    X_2 = Conv2D(100, kernel_size=(2,20), strides=(1,1), activation='relu')(X_input)
    X_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_2)
    X_2 = Reshape((299,100,1))(X_2)
    
    X_2_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1), activation='relu')(X_2)
    X_2_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_1)
    X_2_1 = MaxPooling2D(pool_size=(299,1), padding='valid')(X_2_1)
    X_2_1 = Flatten()(X_2_1)
    X_2_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1), activation='relu')(X_2)
    X_2_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_2)
    X_2_2 = MaxPooling2D(pool_size=(298,1), padding='valid')(X_2_2)
    X_2_2 = Flatten()(X_2_2)
    X_2_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1), activation='relu')(X_2)
    X_2_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_3)
    X_2_3 = MaxPooling2D(pool_size=(297,1),padding='valid')(X_2_3)
    X_2_3 = Flatten()(X_2_3)
    
    X_2 = tf.keras.layers.Concatenate(axis=-1)([X_2_1,X_2_2])
    X_2 = tf.keras.layers.Concatenate(axis=-1)([X_2,X_2_3])
    
    X_3 = Conv2D(100, kernel_size=(3,20), strides=(1,1),activation='relu')(X_input)
    X_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_3)
    X_3 = Reshape((298,100,1))(X_3)
    
    X_3_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1),activation='relu')(X_3)
    X_3_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_1)
    X_3_1 = MaxPooling2D(pool_size=(298,1),padding='valid')(X_3_1)
    X_3_1 = Flatten()(X_3_1)
    X_3_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1),activation='relu')(X_3)
    X_3_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_2)
    X_3_2 = MaxPooling2D(pool_size=(297,1),padding='valid')(X_3_2)
    X_3_2 = Flatten()(X_3_2)
    X_3_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1),activation='relu')(X_3)
    X_3_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_3)
    X_3_3 = MaxPooling2D(pool_size=(296,1),padding='valid')(X_3_3)
    X_3_3 = Flatten()(X_3_3)
    
    X_3 = tf.keras.layers.Concatenate(axis=-1)([X_3_1,X_3_2])
    X_3 = tf.keras.layers.Concatenate(axis=-1)([X_3,X_3_3])
    
    
    X = tf.keras.layers.Concatenate(axis=-1)([X_1,X_2])
    X = tf.keras.layers.Concatenate(axis=-1)([X,X_3])
    
    X = Dropout(0.6)(X)
    X = Dense(300, activation='relu')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    
    
    X = Dropout(0.4)(X)
    X = Dense(100, activation='relu')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)


    X = Dropout(0.4)(X)
    Y = Dense(1, activation='sigmoid')(X)
    model = Model(inputs = X_input, outputs = Y, name='CNN_Model')
    
    logger.info(model.summary())
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name...')
    parser.add_argument('--project', help='project name', required=True)

    args = parser.parse_args()
    
    logger = get_logger('../log/train_{}_{}.log'.format(args.project, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    matrix_data_path = Path('../data/matrix/{}'.format(args.project))
    model_path = Path('../model/{}'.format(args.project))
    model_path.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    logger.info('It started at: %s' % start_time)
    
    # K.set_session(get_session(1.0))  # using 40% of total GPU Memory

    tf.compat.v1.keras.backend.set_session(get_session(1.0))
    
    sabd_data_path = '../../SABD/dataset/{}/'.format(args.project)
    hindbr_train_pairs = '../../HINDBR/data/model_training/{}_training_pairs.txt'.format(args.project)
    matrix_data_path = '../data/matrix/{}/'.format(args.project)
    
    data_train, label_train = represent_training_pairs(
        train_pairs=hindbr_train_pairs, \
        database_path=sabd_data_path + '{}.json'.format(args.project), \
        matrix_data_path=matrix_data_path
    )
    
    label_train = label_train.astype(int)

    index = [i for i in range(len(data_train))]
    # random.seed(42)
    # random.shuffle(index)
    data_train = data_train[index]
    label_train = label_train[index]

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

    dccnnModel = DCCNN_Model((300, 20, 2))
    dccnnModel.compile(
        optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy']
    )
    
    dccnnModel.fit(
        x = data_train, 
        y = label_train,
        batch_size = 64, 
        epochs = 100,
        validation_split = 0.2, 
        callbacks = [early_stopping],
        shuffle = True
    )
    
    end_time = datetime.now()
    
    logger.info('training after: {}'.format(end_time - start_time))
    
    for i in range(1, 20):
        if not os.path.exists(model_path / 'dccnn_{}.h5'.format(i)):
            model_save_name = model_path / 'dccnn_{}.h5'.format(i)
            break

    dccnnModel.save(model_save_name)

    # K.clear_session()
    tf.compat.v1.keras.backend.clear_session()
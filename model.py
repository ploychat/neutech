import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import time 
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
from numbers_parser import Document
from readnumber import *
from eeg_open import *

def AV_Layer_Input( input_shape_p=None,     # Shape tuple (not including the batch axis), or TensorShape instance (not including the batch axis).
                    batch_size_p=None,      # Optional input batch size (integer or None).
                    dtype_p=None,           # Datatype of the input.
                    input_tensor_p=None,    # Optional tensor to use as layer input instead of creating a placeholder.
                    sparse_p=False,         # Boolean, whether the placeholder created is meant to be sparse.
                    name_p=None,            # Name of the layer (string).
                    ragged_p=False):         # Boolean, whether the placeholder created is meant to be ragged. In this case, values of 'None' in the 'shape' argument 
                                            # represent ragged dimensions. For more information about RaggedTensors, see https://www.tensorflow.org/guide/ragged_tensors.

    # return keras.layers.InputLayer( input_shape=input_shape_p, 
    #                                 batch_size=batch_size_p, 
    #                                 dtype=dtype_p, 
    #                                 input_tensor=input_tensor_p, 
    #                                 sparse=sparse_p,
    #                                 name=name_p, 
    #                                 ragged=ragged_p)

    return keras.Input( shape=input_shape_p,
                        batch_size=batch_size_p,
                        name=name_p,
                        dtype=dtype_p,
                        sparse=sparse_p,
                        tensor=None,
                        ragged=ragged_p)

def AV_Layer_Flatten(   data_format_p=None  # A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
                                            # The purpose of this argument is to preserve weight ordering when switching a model from one data format 
                                            # to another. channels_last corresponds to inputs with shape (batch, ..., channels) while channels_first 
                                            # corresponds to inputs with shape (batch, channels, ...). It defaults to the image_data_format value found 
                                            # in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
                        ):

    return keras.layers.Flatten(data_format=data_format_p)

def AV_Layer_Dense( units_p,                                # Positive integer, dimensionality of the output space.
                    activation_p=None,                      # Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
                    use_bias_p=True,                        # Boolean, whether the layer uses a bias vector.
                    kernel_initializer_p='glorot_uniform',  # Initializer for the kernel weights matrix (see initializers).
                    bias_initializer_p='zeros',             # Initializer for the bias vector (see initializers).
                    kernel_regularizer_p=None,              # Regularizer function applied to the kernel weights matrix (see regularizer).
                    bias_regularizer_p=None,                # Regularizer function applied to the bias vector (see regularizer).
                    activity_regularizer_p=None,            # Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
                    kernel_constraint_p=None,               # Constraint function applied to the kernel weights matrix (see constraints).
                    bias_constraint_p=None                  # Constraint function applied to the bias vector (see constraints).
                    ):

    return keras.layers.Dense(  units=units_p, 
                                activation=activation_p, 
                                use_bias=use_bias_p, 
                                kernel_initializer=kernel_initializer_p, 
                                bias_initializer=bias_initializer_p, 
                                kernel_regularizer=kernel_regularizer_p, 
                                bias_regularizer=bias_regularizer_p, 
                                activity_regularizer=activity_regularizer_p, 
                                kernel_constraint=kernel_constraint_p, 
                                bias_constraint=bias_constraint_p)
def AV_Layer_Activation(activation_p="relu", name_p=None, ret_kernel_initializer=False):
    """
    See https://mlfromscratch.com/activation-functions-explained/#/
    """

    if not ret_kernel_initializer:
        ## ReLU solves vanishing gradient by defining two values of gradients either 0 or 1 (nothing close to 0!!!). But, it can be trapped in the dead state (0 gradient).
        ## The way out of the dead state is leaky relu. 
        ## If we want to use Dropout for SELU, we shall use AlphaDropout (see https://mlfromscratch.com/activation-functions-explained/#/)  

        if activation_p == "square":
            return keras.layers.Activation(square, name=name_p)
        elif activation_p == "safe_log":
            return keras.layers.Activation(safe_log, name=name_p)
        elif activation_p == "swish":
            return keras.layers.Activation(swish, name=name_p)                        
        elif activation_p == "leaky_relu":        
            return keras.layers.LeakyReLU(alpha=0.3)
        else:
            return keras.layers.Activation(activation=activation_p, name=name_p)
    else:
        ## See https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc
        ## Assume the output of the previous layer is i1, ..., iN. The outputs are given to the next layer and yield o1, ..., oM (after activation). We want o1, ..., oM all together forms the normal distribution (zero mean, unit std).
        ## If o1, ..., oM follows the normal distribution, the output of the forward layers will not blow up and vanish.
        ## It also means that the input of the deep network should be normalized before.

        if activation_p == "selu": 
            return "lecun_normal"
        elif (activation_p == "relu"):
            ## See https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc
            ## Problem of using he_normal
            ## 1. After relu, the mean of the output is not 0 because relu removes all negative inputs.
            ## 2. It works only relu.
            ## 3. STD of the outputs is close to 1 (not exactly 1).
            return "he_normal" 
        elif (activation_p == "swish") or (activation_p == "elu") or (activation_p == "leaky_relu"):
            return "he_normal"
        else:
            return "glorot_uniform"

def AV_Machine_Conv1D_VGG16(TS_LENGTH_p, TS_RGB_p, nb_classes_p=2, model_params_p=""):
    """
    Trainable params: 165,726,018
    """

    desc_l = ""

    def VGG16(input_tensor=None):
        """Instantiates the VGG16 architecture.

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

        # Arguments
            input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """

        # Block 1
        x = layers.Conv1D(64, 3, padding='same', name='block1_conv1')(input_tensor)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(64, 3, padding='same', name='block1_conv2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, name='block1_pool')(x)

        # Block 2
        x = layers.Conv1D(128, 3, padding='same', name='block2_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(128, 3, padding='same', name='block2_conv2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, name='block2_pool')(x)

        # Block 3
        x = layers.Conv1D(256, 3, padding='same', name='block3_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(256, 3, padding='same', name='block3_conv2')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(256, 3, padding='same', name='block3_conv3')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, name='block3_pool')(x)

        # Block 4
        x = layers.Conv1D(512, 3, padding='same', name='block4_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(512, 3, padding='same', name='block4_conv2')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(512, 3, padding='same', name='block4_conv3')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, name='block4_pool')(x)

        # Block 5
        x = layers.Conv1D(512, 3, padding='same', name='block5_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(512, 3, padding='same', name='block5_conv2')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(512, 3, padding='same', name='block5_conv3')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, strides=2, name='block5_pool')(x)

        # Create model.
        model = keras.models.Model(inputs=input_tensor, outputs=x, name='vgg16')

        return model
    

    x_l = AV_Layer_Input(input_shape_p=(TS_LENGTH_p, TS_RGB_p))

    y_l = VGG16(input_tensor=x_l).output

    y_l = AV_Layer_Flatten()(y_l)

    y_l = AV_Layer_Dense(units_p=4096)(y_l)
    y_l = AV_Layer_Activation(activation_p="relu")(y_l)

    y_l = AV_Layer_Dense(units_p=4096)(y_l)
    y_l = AV_Layer_Activation(activation_p="relu")(y_l)
    
    y_l = AV_Layer_Dense(units_p=nb_classes_p)(y_l)    
    y_l = AV_Layer_Activation(activation_p="softmax")(y_l)

    return keras.models.Model(inputs=x_l, outputs=y_l), desc_l

model_l, _ = AV_Machine_Conv1D_VGG16(TS_LENGTH_p=256, TS_RGB_p=1*4, nb_classes_p=2, model_params_p="")

def AV_Model_compile(model_p, 
                        optimizer_p,                # String (name of optimizer) or optimizer instance. See optimizers. 
                        loss_p=None,                # String (name of objective function) or objective function or Loss instance. See losses. 
                                                    # If the model has multiple outputs, you can use a different loss on each output by passing 
                                                    # a dictionary or a list of losses. The loss value that will be minimized by the model will then 
                                                    # be the sum of all individual losses.
                        metrics_p=None,             # List of metrics to be evaluated by the model during training and testing. Typically you will 
                                                    # use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output 
                                                    # model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. 
                                                    # You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], 
                                                    # ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']].
                        loss_weights_p=None,        # Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss 
                                                    # contributions of different model outputs. The loss value that will be minimized by the model will 
                                                    # then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. 
                                                    # If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected 
                                                    # to map output names (strings) to scalar coefficients. 
                        sample_weight_mode_p=None,  # If you need to do timestep-wise sample weighting (2D weights), set this to "temporal". 
                                                    # None defaults to sample-wise weights (1D). If the model has multiple outputs, you can use 
                                                    # a different sample_weight_mode on each output by passing a dictionary or a list of modes.
                        weighted_metrics_p=None,    # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                        target_tensors_p=None       # By default, Keras will create placeholders for the model's target, which will be fed with the 
                                                    # target data during training. If instead you would like to use your own target tensors (in turn, 
                                                    # Keras will not expect external Numpy data for these targets at training time), you can specify 
                                                    # them via the target_tensors argument. It can be a single tensor (for a single-output model), 
                                                    # a list of tensors, or a dict mapping output names to target tensors.
                        ):
    """
    Configures the model for training.
    """

    model_p.compile(optimizer=optimizer_p, 
                    loss=loss_p, 
                    metrics=metrics_p, 
                    loss_weights=loss_weights_p, 
                    sample_weight_mode=sample_weight_mode_p, 
                    weighted_metrics=weighted_metrics_p, 
                    target_tensors=target_tensors_p)

AV_Model_compile(model_p=putten_model_l, optimizer_p="adam", loss_p="categorical_crossentropy", metrics_p=['accuracy'])

        # AV_Model_summarize(model_p=putten_model_l)

early_stopping_l = EarlyStopping(patience=0, verbose=1)


history_l = AV_Model_train_allbatches(  model_p=putten_model_l,
                                                x_p=x_trains_l,                       # Input data. It could be:
                                                                                #   A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                #   A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                #   A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                #   None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                y_p=Y_train_l,                       # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                batch_size_p=batch_size_l,              # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                epochs_p=epochs_l,                     # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                #callbacks_p=[early_stopping_l],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                )        


score_l = putten_model_l.evaluate(x_tests_l, Y_test_l, verbose=0)
print('Test score:', score_l[0])
print('Test accuracy:', score_l[1])

def AV_Model_train_allbatches(  model_p,
                                x_p=None,                       # Input data. It could be:
                                                                #   A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                #   A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                #   A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                #   None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                y_p=None,                       # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                batch_size_p=None,              # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                epochs_p=1,                     # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                callbacks_p=None,               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                validation_data_p=None,         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                ):                                
    """
    This method is different from AV_Model_train_batchBybatch() in such a way that if we can load all trainning and validation data inside the memory all at once.
    Another difference is that there is no data augmentation in this function, i.e. we train on the raw data.
    """

    history_l = model_p.fit(x=x_p, 
                            y=y_p, 
                            batch_size=batch_size_p, 
                            epochs=epochs_p, 
                            verbose=verbose_p, 
                            callbacks=callbacks_p, 
                            validation_split=validation_split_p, 
                            validation_data=validation_data_p, 
                            shuffle=shuffle_p, 
                            class_weight=class_weight_p, 
                            sample_weight=sample_weight_p, 
                            initial_epoch=initial_epoch_p, 
                            steps_per_epoch=steps_per_epoch_p, 
                            validation_steps=validation_steps_p, 
                            validation_freq=validation_freq_p, 
                            max_queue_size=max_queue_size_p, 
                            workers=workers_p, 
                            use_multiprocessing=use_multiprocessing_p)

    return history_l


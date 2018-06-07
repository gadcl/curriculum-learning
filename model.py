from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, \
    Input
import json
import os


class Model:
    def __init__(self, dataset):
        if K.image_data_format() == 'channels_first':
            self.input_shape = (dataset.num_channels, dataset.height, dataset.width)
        else:
            self.input_shape = (dataset.height, dataset.width, dataset.num_channels)
        self.num_classes = dataset.num_classes

    def create_model(self, cfg_file, net_type, **kwargs):
        with open(cfg_file) as json_file:
            arch_specs = json.load(json_file)
        arch = arch_specs[net_type]

        l2_reg = regularizers.l2(0.0)
        l2_bias_reg = regularizers.l2(0.0)
        dropout_rate = [0.0,0.0]
        batch_norm = False
        activation = 'elu'
        if 'activation' in kwargs:
            activation =kwargs['activation']
        if 'reg_factor' in kwargs:
            l2_reg = regularizers.l2(kwargs['reg_factor'])
        if 'bias_reg_factor' in kwargs:
            l2_bias_reg = regularizers.l2(kwargs['bias_reg_factor'])
        if 'dropout_rate' in kwargs:
            dropout_rate = kwargs['dropout_rate']
        if 'batch_norm' in kwargs:
            batch_norm =kwargs['batch_norm']

        #build model
        # input image dimensions
        x = input_1 = Input(shape=self.input_shape)

        for layer in range(arch['num_block_layers']):
            x = Conv2D(filters=arch['filters'][layer], kernel_size=arch['kernel_size'][layer], padding='same', kernel_regularizer=l2_reg,
                       bias_regularizer=l2_bias_reg)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)
            x = Conv2D(filters=arch['filters'][layer], kernel_size=arch['kernel_size'][layer], padding='same', kernel_regularizer=l2_reg,
                       bias_regularizer=l2_bias_reg)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(rate=dropout_rate[0])(x)

        x = Flatten()(x)
        for layer in range(arch['num_dense_layers']):
            x = Dense(units=arch['units'][layer], kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)
            x = Dropout(rate=dropout_rate[1])(x)

        x = Dense(units=self.num_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation='softmax')(x)

        model = Model(inputs=[input_1], outputs=[x])
        model.summary()
        return model


    def generate_init_weights(self, cfg_file, net_type, num_inits, dest_path):
        destination = os.path.join(dest_path, net_type + '/')
        if not os.path.exists(destination):
            os.makedirs(destination)
        print(net_type)
        for j in range(num_inits):
            try:
                model = self.create_model(cfg_file, net_type)
                model.save(os.path.join(destination, 'model_init_{0}.h5'.format(j)))
                print(j)
            except:
                pass

        return None


#
# def create_model(net_type='large',n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.5,
#                            reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
#
#     if net_type == 'large':
#         return _build_model_large(n_classes=n_classes,activation = activation,dropout_1_rate=dropout_1_rate,
#                                   dropout_2_rate = dropout_2_rate,reg_factor = reg_factor,
#                                   bias_reg_factor=bias_reg_factor,batch_norm=batch_norm )
#
#     elif net_type == 'medium':
#         return _build_model_medium(n_classes=n_classes,activation = activation,dropout_1_rate=dropout_1_rate,
#                                   dropout_2_rate = dropout_2_rate,reg_factor = reg_factor,
#                                   bias_reg_factor=bias_reg_factor,batch_norm=batch_norm )
#
#     if net_type == 'small':
#         return _build_model_small(n_classes=n_classes,activation = activation,dropout_1_rate=dropout_1_rate,
#                                   dropout_2_rate = dropout_2_rate,reg_factor = reg_factor,
#                                   bias_reg_factor=bias_reg_factor,batch_norm=batch_norm )
#     else:
#         return None
#
#
#
# def _build_model_large(n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.5,
#                            reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
#
#     l2_reg = regularizers.l2(reg_factor) #K.variable(K.cast_to_floatx(reg_factor))
#     l2_bias_reg = None
#     if bias_reg_factor:
#         l2_bias_reg = regularizers.l2(bias_reg_factor) #K.variable(K.cast_to_floatx(bias_reg_factor))
#
#     # input image dimensions
#     h, w, d = 32, 32, 3
#
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, h, w)
#     else:
#         input_shape = (h, w, 3)
#
#     # input image dimensions
#     x = input_1 = Input(shape=input_shape)
#
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Flatten()(x)
#     x = Dense(units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#
#
#     x = Dropout(rate=dropout_2_rate)(x)
#     x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation='softmax')(x)
#
#     model = Model(inputs=[input_1], outputs=[x])
#     # model.l2_reg = l2_reg
#     # model.l2_bias_reg = l2_bias_reg
#     return model
#
#
#
#
#
# def _build_model_medium(n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.5,
#                            reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
#     l2_reg = regularizers.l2(reg_factor)
#     l2_bias_reg = None
#     if bias_reg_factor:
#         regularizers.l2(bias_reg_factor)
#
#
#     # input image dimensions
#     h, w, d = 32, 32, 3
#
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, h, w)
#     else:
#         input_shape = (h, w, 3)
#
#     # input image dimensions
#     x = input_1 = Input(shape=input_shape)
#
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#     x = Flatten()(x)
#     x = Dense(units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#
#
#     x = Dropout(rate=dropout_2_rate)(x)
#     x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation='softmax')(x)
#
#     return Model(inputs=[input_1], outputs=[x])
#
#
#
# def _build_model_small2(n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.25,
#                            reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
#     l2_reg = regularizers.l2(reg_factor)
#     l2_bias_reg = None
#     if bias_reg_factor:
#         regularizers.l2(bias_reg_factor)
#
#
#     # input image dimensions
#     h, w, d = 32, 32, 3
#
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, h, w)
#     else:
#         input_shape = (h, w, 3)
#
#     # input image dimensions
#     x = input_1 = Input(shape=input_shape)
#
#     x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#
#     x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#
#     x = Flatten()(x)
#     x = Dense(units=16, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#
#
#     x = Dropout(rate=dropout_2_rate)(x)
#     x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation='softmax')(x)
#
#     return Model(inputs=[input_1], outputs=[x])
#
#
#
# def _build_model_small(n_classes=5,activation = 'elu',dropout_1_rate=0.25,dropout_2_rate = 0.25,
#                            reg_factor = 200e-4, bias_reg_factor=None,batch_norm=False ):
#     l2_reg = regularizers.l2(reg_factor)
#     l2_bias_reg = None
#     if bias_reg_factor:
#         regularizers.l2(bias_reg_factor)
#
#
#     # input image dimensions
#     h, w, d = 32, 32, 3
#
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, h, w)
#     else:
#         input_shape = (h, w, 3)
#
#     # input image dimensions
#     x = input_1 = Input(shape=input_shape)
#
#     x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     # if batch_norm:
#     #     x = BatchNormalization()(x)
#     # x = Activation(activation=activation)(x)
#     # x = Conv2D(filters=4, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     # if batch_norm:
#     #     x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#
#     x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     # if batch_norm:
#     #     x = BatchNormalization()(x)
#     # x = Activation(activation=activation)(x)
#     # x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     # if batch_norm:
#     #     x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(rate=dropout_1_rate)(x)
#
#
#     x = Flatten()(x)
#     x = Dense(units=8, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation=activation)(x)
#
#
#     x = Dropout(rate=dropout_2_rate)(x)
#     x = Dense(units=n_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Activation(activation='softmax')(x)
#
#     return Model(inputs=[input_1], outputs=[x])
#
#

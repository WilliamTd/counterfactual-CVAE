import keras
import numpy as np

def get_keras_encoder(
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        n_filters_c2,
        size_filters_c2,
        n_filters_c3,
        size_filters_c3,
        latent_dim,
    ):
    """
    Only works with classic CNN encoder with 3 layers
    """
    def enc_layer(x, n_filters, size_filters, name, pool=True):
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=size_filters,
            padding="same",
            name=f"{name}_conv",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = keras.layers.LeakyReLU(alpha=0.01, name=f"{name}_leakyrelu")(x)
        if pool :
            x = keras.layers.MaxPool1D(
                pool_size=2,
                strides=2,
                padding="valid",
                name=f"{name}_maxpool"
            )(x)
            x = keras.layers.Dropout(rate=0.2, name=f"{name}_dropout")(x)
        return x

    inputs = keras.layers.Input(shape=(length_ts, input_channels))
    x = enc_layer(inputs, n_filters_c1, size_filters_c1, name="enc1")
    x = enc_layer(x, n_filters_c2, size_filters_c2, name="enc2")
    x = enc_layer(x, n_filters_c3, size_filters_c3, name="enc3", pool=False)
    x = keras.layers.Permute((2,1))(x) # here to have the same order than pytorch after the flatten
    x = keras.layers.Flatten(name='flatten')(x)
    mu = keras.layers.Dense(latent_dim, name='dense_mu')(x)
    log_var = keras.layers.Dense(latent_dim, name='dense_log_var')(x)
    model = keras.models.Model(inputs, [mu, log_var], name='keras_encoder')
    return model

def get_keras_decoder(
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        n_filters_c2,
        size_filters_c2,
        n_filters_c3,
        size_filters_c3,
        latent_dim,
    ):
    """
    Only works with classic CNN decoder with 3 layers and upsample+conv instead of convtranspose
    """
    bottleneck_length = length_ts // 2 ** 2

    def upsample_layer(x, steps, features, ratio=2, interpolation="nearest", name=""):
        assert interpolation in ["linear", "nearest"], "linear and nearest interpolation supported"
        if interpolation == "linear" :
            interpolation = "bi"+interpolation

        x = keras.layers.Reshape((steps,1,features), name=f"{name}_reshape0")(x)
        x = keras.layers.UpSampling2D(size=(ratio, 1), interpolation=interpolation,name=f"{name}_up2d")(x)
        x = keras.layers.Reshape((steps*ratio,features), name=f"{name}_reshape1")(x)
        return x

    def dec_layer(x, steps, features, n_filters, size_filters, name):
        x = upsample_layer(x, steps, features, ratio=2, interpolation="nearest", name = f"{name}_up")
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=size_filters,
            padding="same",
            name=f"{name}_conv"
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = keras.layers.LeakyReLU(alpha=0.01, name=f"{name}_leakyrelu")(x)
        x = keras.layers.Dropout(rate=0.2, name=f"{name}_dropout")(x)
        return x

    inputs = keras.layers.Input(shape=(latent_dim))
    x = keras.layers.Dense(bottleneck_length * n_filters_c3, name='denseinput')(inputs)
    x = keras.layers.Reshape((n_filters_c3, bottleneck_length), name='reshape')(x)
    x = keras.layers.Permute((2,1))(x)
    x = keras.layers.Conv1D(
        filters=n_filters_c2,
        kernel_size=size_filters_c2,
        padding="same",
        name="conv0"
    )(x)
    x = keras.layers.BatchNormalization(name="conv0_bn")(x)
    x = keras.layers.LeakyReLU(alpha=0.01, name="conv0_leakyrelu")(x)
    x = keras.layers.Dropout(rate=0.2, name="conv0_dropout")(x)
    x = dec_layer(x, steps=bottleneck_length, features=n_filters_c2, n_filters=n_filters_c1, size_filters=size_filters_c1, name="dec1")
    x = dec_layer(x, steps=2*bottleneck_length, features=n_filters_c1, n_filters=n_filters_c1, size_filters=size_filters_c1, name="dec2")
    reconstruction = keras.layers.Conv1D(
        filters=input_channels,
        kernel_size=1,
        padding="same",
        name="convfinallayer"
    )(x)
    model = keras.models.Model(inputs, reconstruction, name='keras_decoder')
    return model


def get_keras_classifier(
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        n_filters_c2,
        size_filters_c2,
        n_filters_c3,
        size_filters_c3,
        n_fc1,
        n_fc2,
        n_output,
    ):

    def conv_block(x, n_filters, size_filters, pool=True, name=""):
        x = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=size_filters,
            padding="same",
            name=f"{name}_conv",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = keras.layers.LeakyReLU(alpha=0.01, name=f"{name}_leakyrelu")(x)
        if pool :
            x = keras.layers.MaxPool1D(
                pool_size=2,
                strides=2,
                padding="valid",
                name=f"{name}_maxpool"
            )(x)
            x = keras.layers.Dropout(rate=0.2, name=f"{name}_dropout")(x)
        return x

    def fc_block(x, n_neurons, name=""):
        x = keras.layers.Dense(n_neurons, name=f"{name}_dense")(x)
        x = keras.layers.LeakyReLU(alpha=0.01, name=f"{name}_leakyrelu")(x)
        x = keras.layers.Dropout(rate=0.2, name=f"{name}_dropout")(x)
        return x

    inputs = keras.layers.Input(shape=(length_ts, input_channels))
    x = conv_block(inputs, n_filters_c1, size_filters_c1, name="convblock1")
    x = conv_block(x, n_filters_c2, size_filters_c2, name="convblock2")
    x = conv_block(x, n_filters_c3, size_filters_c3, name="convblock3")
    x = keras.layers.Permute((2,1))(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = fc_block(x, n_fc1, name="fc1")
    x = fc_block(x, n_fc2, name="fc2")
    pred = keras.layers.Dense(n_output, name="output")(x)
    model = keras.models.Model(inputs, pred, name="keras_classifier")
    return model


def transfer_pytorch2keras_encoder(keras_model, torch_model):
    # enc_1
    ## conv
    keras_model.layers[1].set_weights(
        [np.transpose(torch_model.enc_layer1[0].weight.detach().numpy(), (2,1,0)),
         torch_model.enc_layer1[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.enc_layer1[1].weight.detach().numpy()
    bias = torch_model.enc_layer1[1].bias.detach().numpy()
    running_mean =  torch_model.enc_layer1[1].running_mean.detach().numpy()
    running_var =  torch_model.enc_layer1[1].running_var.detach().numpy()
    keras_model.layers[2].set_weights([weights, bias, running_mean, running_var])

    # enc_2
    ## conv
    keras_model.layers[6].set_weights(
        [np.transpose(torch_model.enc_layer2[0].weight.detach().numpy(), (2,1,0)),
         torch_model.enc_layer2[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.enc_layer2[1].weight.detach().numpy()
    bias = torch_model.enc_layer2[1].bias.detach().numpy()
    running_mean =  torch_model.enc_layer2[1].running_mean.detach().numpy()
    running_var =  torch_model.enc_layer2[1].running_var.detach().numpy()
    keras_model.layers[7].set_weights([weights, bias, running_mean, running_var])

    # enc_3
    ## conv
    keras_model.layers[11].set_weights(
        [np.transpose(torch_model.enc_layer3[0].weight.detach().numpy(), (2,1,0)),
         torch_model.enc_layer3[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.enc_layer3[1].weight.detach().numpy()
    bias = torch_model.enc_layer3[1].bias.detach().numpy()
    running_mean =  torch_model.enc_layer3[1].running_mean.detach().numpy()
    running_var =  torch_model.enc_layer3[1].running_var.detach().numpy()
    keras_model.layers[12].set_weights([weights, bias, running_mean, running_var])

    # fc_mu
    keras_model.layers[16].set_weights(
        [torch_model.fc_mu.weight.detach().numpy().T,
        torch_model.fc_mu.bias.detach().numpy()]
    )
    # fc_var
    keras_model.layers[17].set_weights(
        [torch_model.fc_var.weight.detach().numpy().T,
        torch_model.fc_var.bias.detach().numpy()]
    )
    return keras_model


def transfer_pytorch2keras_decoder(keras_model, torch_model):
    # linear 1
    keras_model.layers[1].set_weights(
        [torch_model.dec_input.weight.detach().numpy().T,
        torch_model.dec_input.bias.detach().numpy()]
    )
    # dec 0
    ## conv
    keras_model.layers[4].set_weights(
        [np.transpose(torch_model.dec_layer0[0].weight.detach().numpy(), (2,1,0)),
         torch_model.dec_layer0[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.dec_layer0[1].weight.detach().numpy()
    bias = torch_model.dec_layer0[1].bias.detach().numpy()
    running_mean =  torch_model.dec_layer0[1].running_mean.detach().numpy()
    running_var =  torch_model.dec_layer0[1].running_var.detach().numpy()
    keras_model.layers[5].set_weights([weights, bias, running_mean, running_var])

    # dec1
    ## conv
    keras_model.layers[11].set_weights(
        [np.transpose(torch_model.dec_layer1[1].weight.detach().numpy(), (2,1,0)),
         torch_model.dec_layer1[1].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.dec_layer1[2].weight.detach().numpy()
    bias = torch_model.dec_layer1[2].bias.detach().numpy()
    running_mean =  torch_model.dec_layer1[2].running_mean.detach().numpy()
    running_var =  torch_model.dec_layer1[2].running_var.detach().numpy()
    keras_model.layers[12].set_weights([weights, bias, running_mean, running_var])

    # dec final
    ## conv
    keras_model.layers[18].set_weights(
        [np.transpose(torch_model.dec_final_layer[1].weight.detach().numpy(), (2,1,0)),
         torch_model.dec_final_layer[1].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.dec_final_layer[2].weight.detach().numpy()
    bias = torch_model.dec_final_layer[2].bias.detach().numpy()
    running_mean =  torch_model.dec_final_layer[2].running_mean.detach().numpy()
    running_var =  torch_model.dec_final_layer[2].running_var.detach().numpy()
    keras_model.layers[19].set_weights([weights, bias, running_mean, running_var])

    # last conv
    keras_model.layers[22].set_weights(
        [np.transpose(torch_model.dec_final_layer[5].weight.detach().numpy(), (2,1,0)),
         torch_model.dec_final_layer[5].bias.detach().numpy()]
    )
    return keras_model


def transfer_pytorch2keras_classifier(keras_model, torch_model):
    # conv_layer1
    ## conv
    keras_model.layers[1].set_weights(
        [np.transpose(torch_model.conv_layer1[0].weight.detach().numpy(), (2,1,0)),
         torch_model.conv_layer1[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.conv_layer1[1].weight.detach().numpy()
    bias = torch_model.conv_layer1[1].bias.detach().numpy()
    running_mean =  torch_model.conv_layer1[1].running_mean.detach().numpy()
    running_var =  torch_model.conv_layer1[1].running_var.detach().numpy()
    keras_model.layers[2].set_weights([weights, bias, running_mean, running_var])

    # conv_layer2
    ## conv
    keras_model.layers[6].set_weights(
        [np.transpose(torch_model.conv_layer2[0].weight.detach().numpy(), (2,1,0)),
         torch_model.conv_layer2[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.conv_layer2[1].weight.detach().numpy()
    bias = torch_model.conv_layer2[1].bias.detach().numpy()
    running_mean =  torch_model.conv_layer2[1].running_mean.detach().numpy()
    running_var =  torch_model.conv_layer2[1].running_var.detach().numpy()
    keras_model.layers[7].set_weights([weights, bias, running_mean, running_var])

    # conv_layer3
    ## conv
    keras_model.layers[11].set_weights(
        [np.transpose(torch_model.conv_layer3[0].weight.detach().numpy(), (2,1,0)),
         torch_model.conv_layer3[0].bias.detach().numpy()]
    )
    ## bn
    weights = torch_model.conv_layer3[1].weight.detach().numpy()
    bias = torch_model.conv_layer3[1].bias.detach().numpy()
    running_mean =  torch_model.conv_layer3[1].running_mean.detach().numpy()
    running_var =  torch_model.conv_layer3[1].running_var.detach().numpy()
    keras_model.layers[12].set_weights([weights, bias, running_mean, running_var])

    # fc1
    keras_model.layers[18].set_weights(
        [torch_model.fc1_layer[0].weight.detach().numpy().T,
        torch_model.fc1_layer[0].bias.detach().numpy()]
    )
    # fc2
    keras_model.layers[21].set_weights(
        [torch_model.fc2_layer[0].weight.detach().numpy().T,
        torch_model.fc2_layer[0].bias.detach().numpy()]
    )
    # output
    keras_model.layers[24].set_weights(
        [torch_model.output_layer.weight.detach().numpy().T,
        torch_model.output_layer.bias.detach().numpy()]
    )
    return keras_model

import ml_collections


def get_3DReg_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (10, 16, 16)})
    # config.patches.grid = (10, 10, 10)
    config.hidden_size = 252
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.patch_size = 10

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 0
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 16, 16)
    # config.skip_channels = (32, 32, 32, 32, 16)
    config.skip_channels = (0, 0, 0, 0, 0)  # TODO: fix important!!!
    config.n_dims = 1
    config.n_skip = 5  # TODO: fix important!!!
    return config

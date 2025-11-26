import keras


def create_model_full():
    inputs = keras.layers.Input(shape=(30, 2))

    # CNN
    x = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # LSTM + Attention
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_short_history():
    inputs = keras.layers.Input(shape=(20, 2))

    # CNN
    x = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # LSTM + Attention
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_only_lstm():
    inputs = keras.layers.Input(shape=(30, 2))

    # LSTM
    x = keras.layers.LSTM(256, return_sequences=True)(inputs)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_cnn_lstm():
    inputs = keras.layers.Input(shape=(30, 2))

    # CNN
    x = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # LSTM
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_lstm_attention():
    inputs = keras.layers.Input(shape=(30, 2))

    # LSTM + Attention
    x = keras.layers.LSTM(256, return_sequences=True)(inputs)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_bidirectional_lstm():
    inputs = keras.layers.Input(shape=(30, 2))

    # Bidirectional LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(inputs)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_gru():
    inputs = keras.layers.Input(shape=(30, 2))

    # GRU
    x = keras.layers.GRU(256, return_sequences=True)(inputs)
    x = keras.layers.GRU(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_bidirectional_gru():
    inputs = keras.layers.Input(shape=(30, 2))

    # Bidirectional GRU
    x = keras.layers.Bidirectional(
        keras.layers.GRU(128, return_sequences=True)
    )(inputs)
    x = keras.layers.GRU(128)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model


def create_model_transformer_encoder():
    inputs = keras.layers.Input(shape=(30, 2))

    d_model = 128

    x = keras.layers.Dense(d_model)(inputs)

    attn_out = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = keras.layers.Add()([x, attn_out])
    x = keras.layers.LayerNormalization()(x)

    ff = keras.layers.Dense(256, activation='relu')(x)
    ff = keras.layers.Dense(d_model)(ff)
    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dropout(0.3)(x)

    # Output
    outputs = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.L2())(x)
    outputs = keras.layers.Dense(60)(outputs)
    outputs = keras.layers.Reshape((20, 3))(outputs)

    model = keras.Model(inputs, outputs)

    return model

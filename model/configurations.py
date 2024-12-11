import keras
# Training

WINDOW_SIZE = 30
N_CHANNELS = 1

LOSS_FUNCTION = keras.losses.BinaryCrossentropy(from_logits=False)

OPTIMIZER = keras.optimizers.Adadelta()

BATCH_SIZE = 32

POSITIVE_PER_EPOCH = BATCH_SIZE // 2 # 16 positive and 16 negative per batch

# optimal params resulting from Bayesian optimization procedure (given by paper)
MODEL_PARAMS = {
    "abdores": {
        "n1": 100,
        "n2": 50,
        "p1": 0.5,
        "p2": 0.5
    },
    "thorres": {
        "n1": 100,
        "n2": 50,
        "p1": 0.5,
        "p2": 0.5
    },
    "EDR": {
        "n1": 50,
        "n2": 20,
        "p1": 0.14,
        "p2": 0.27
    },
}


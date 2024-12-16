import keras
# Training
SAMPLING_RATE = 5
WINDOW_SIZE = 30 * SAMPLING_RATE
N_CHANNELS = 3

LOSS_FUNCTION = keras.losses.BinaryCrossentropy(from_logits=False)

OPTIMIZER = keras.optimizers.Adadelta()

BATCH_SIZE = 32

POSITIVE_PER_EPOCH = BATCH_SIZE // 2 # 16 positive and 16 negative per batch

# optimal params resulting from Bayesian optimization procedure (given by paper)
MODEL_PARAMS = {
    "thorres": {
        "num": 0,
        "n1": 100,
        "n2": 50,
        "p1": 0.5,
        "p2": 0.5
    },
    "abdores": {
        "num": 1,
        "n1": 100,
        "n2": 50,
        "p1": 0.5,
        "p2": 0.5
    },
    "EDR": {
        "num": 2,
        "n1": 50,
        "n2": 20,
        "p1": 0.14,
        "p2": 0.27
    },
}


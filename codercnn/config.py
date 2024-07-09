class Config:
    NAME = "custom_config"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2  # Background + Object
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

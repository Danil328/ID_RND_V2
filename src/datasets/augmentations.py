from albumentations import (RandomRotate90, Normalize, RandomBrightnessContrast, GaussNoise, RandomCrop, Resize,
                            Flip, OneOf, Compose)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentations(steps_probas, height=600, width=600):
    return Compose([
        Resize(height=height, width=width),
        Compose([RandomCrop(int(0.9 * height), int(0.9 * width)), Resize(height, width)], p=steps_probas[0]),
        OneOf([RandomRotate90(), Flip()], p=steps_probas[1]),
        OneOf([RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1), GaussNoise()], p=steps_probas[2]),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ], p=1.0)


def get_test_augmentations(height=600, width=600):
    return Compose([
        Resize(height=height, width=width),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ], p=1.0)

import torchvision.transforms as transforms


def data_augmentation():
    """
    The function `data_augmentation` returns a transformation pipeline for augmenting image data.
    :return: The function `data_augmentation` returns a `transform` object.
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

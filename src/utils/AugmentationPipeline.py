import kornia.augmentation as K


class AugmentationPipeline:
    def __init__(self):
        self.aff = K.RandomAffine(360, return_transform=True, same_on_batch=True)
        self.cj = K.ColorJitter(0.0, 0.0, 0.0, 0.0)

    def forward(self, image):
        image, transform = self.aff(image)
        image, transform = self.jit((image, transform))
        return image

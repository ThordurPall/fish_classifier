import kornia.augmentation as K
import kornia.enhance as E


class AugmentationPipeline:
    def __init__(self):
        self.aff = K.RandomAffine(360, return_transform=True, same_on_batch=True)
        self.cj = K.ColorJitter(0.1, 0.1, 0.1, 0.1)
        self.brightness = E.AdjustBrightness(0.0)

    def forward(self, image):
        image = self.brightness(image)
        image, transform = self.aff(image)
        image, transform = self.cj((image, transform))
        return image

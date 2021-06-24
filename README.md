Fish Classifier
==============================

This project aims to create a neural network based classifier that can successfully classify certain seafood type species. The network architecture will maintain a basic structure consisting of a couple of convolutional and pool layers followed by fully connected layers at the end. The <a target="_blank" href="https://www.kaggle.com/crowww/a-large-scale-fish-dataset">dataset</a> that will be used for the project consists of a total of 430 images of 9 species of fish types [<a target="_blank" href="https://ieeexplore.ieee.org/abstract/document/9259867">1</a>]. The images have three channels and resolution of either 2832 x 2128 or 1024 x 768. The project uses the Kornia framework in an attempt to improve generalization by introducing different data augmentations prior to model training.


<p><small>Note that the project is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.

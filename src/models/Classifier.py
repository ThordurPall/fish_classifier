import torch.nn as nn
import torch.nn.functional as F

## CONVOLUTIONAL LAYER NEURAL NETWORK


def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


class Classifier(nn.Module):
    def __init__(
        self,
        num_classes,
        filter1_in,
        filter1_out,
        filter2_out,
        filter3_out,
        height,
        width,
        pad,
        stride,
        kernel,
        pool,
        fc_1,
        fc_2,
        activation,
        dropout_p=0.25,
    ):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.filter1_in = filter1_in
        self.filter1_out = filter1_out
        self.filter2_out = filter2_out
        self.filter3_out = filter3_out
        self.height = height
        self.width = width
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
        self.pool = pool
        self.fc_1 = fc_1
        self.fc_2 = fc_2
        self.activation = activation
        self.also_return_features = False

        # First convolution
        self.conv1 = nn.Conv2d(self.filter1_in, self.filter1_out, self.kernel)
        # evaluating image dimensions after first connvolution
        self.conv1_out_height = compute_conv_dim(
            self.height, self.kernel, self.pad, self.stride
        )
        self.conv1_out_width = compute_conv_dim(
            self.width, self.kernel, self.pad, self.stride
        )

        # first pooling
        self.pool1 = nn.MaxPool2d(self.pool, self.pool)
        # evaluating image dimensions after first pooling
        self.conv2_out_height = compute_conv_dim(
            self.conv1_out_height, self.pool, self.pad, self.pool
        )
        self.conv2_out_width = compute_conv_dim(
            self.conv1_out_width, self.pool, self.pad, self.pool
        )

        # Second Convolution
        self.conv2 = nn.Conv2d(self.filter1_out, self.filter2_out, self.kernel)
        # evaluating image dimensions after second convolution
        self.conv3_out_height = compute_conv_dim(
            self.conv2_out_height, self.kernel, self.pad, self.stride
        )
        self.conv3_out_width = compute_conv_dim(
            self.conv2_out_width, self.kernel, self.pad, self.stride
        )

        # Second pooling
        self.pool2 = nn.MaxPool2d(self.pool, self.pool)
        # evaluating image dimensions after first pooling
        self.conv4_out_height = compute_conv_dim(
            self.conv3_out_height, self.pool, self.pad, self.pool
        )
        self.conv4_out_width = compute_conv_dim(
            self.conv3_out_width, self.pool, self.pad, self.pool
        )

        # Third Convolution
        self.conv3 = nn.Conv2d(self.filter2_out, self.filter3_out, self.kernel)
        # evaluating image dimensions after second convolution
        self.conv5_out_height = compute_conv_dim(
            self.conv4_out_height, self.kernel, self.pad, self.stride
        )
        self.conv5_out_width = compute_conv_dim(
            self.conv4_out_width, self.kernel, self.pad, self.stride
        )

        self.fc1 = nn.Linear(
            filter3_out * self.conv5_out_height * self.conv5_out_width, fc_1
        )
        self.fc2 = nn.Linear(fc_1, fc_2)
        self.fc3 = nn.Linear(fc_2, self.num_classes)

        # Dropout module with dropout_p drop probability
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):

        # Chack that there are batch, channel, width and height dimensions
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")

        if self.activation == "relu":
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))

            x = x.view(
                -1, self.filter3_out * self.conv5_out_height * self.conv5_out_width
            )
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))

        elif self.activation == "leaky_relu":
            x = self.pool1(F.leaky_relu(self.conv1(x)))
            x = self.pool2(F.leaky_relu(self.conv2(x)))
            x = F.leaky_relu(self.conv3(x))

            x = x.view(
                -1, self.filter3_out * self.conv5_out_height * self.conv5_out_width
            )
            x = self.dropout(F.leaky_relu(self.fc1(x)))
            x = self.dropout(F.leaky_relu(self.fc2(x)))

        else:
            raise ValueError("Activation function not supported")

        # Store the abstract features at this point
        features = x

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        if self.also_return_features:
            return x, features
        return x

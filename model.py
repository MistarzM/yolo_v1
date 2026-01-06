import torch
import torch.nn as nn

"""
            W_in + 2 * Padding - Kernel
    W_out = ---------------------------  + 1
                    Stride
"""
architecture_config = [
    # (kernel_size, number_of_filters, stride, padding)

    # LAYER 1 
    # Input: 448x448x3 (Raw RGB Image)
    # Math: We slide a 7x7 window over the image
    # Stride: 2 (skipping pixels) => shrinking the image by half
    (7, 64, 2, 3), 
    # MAXPOL 
    # Math: It takes the max value of a 2x2 grid. Reduces 224x224 -> 112x112
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # LISTS
    # We do 1x1 conv (compress) -> 3x3 conv (process)
    # This saves computation compared to doing big 3x3 convolution on thick volumes
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3,  512, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # CONVOLUTION
        # Concept: Detects patterns (edges, textures)
        """
            y = Wx + b
        """
        # CRITICAL: bias=False because the very next step is BatchNorm(subtracts the mean)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) 
        # BATCH NORM 
        # Concept: Forces data to have Mean=0 and Variance=1
        """
                      (y - mean)
            z = ------------------------ 
                sqrt(variance + epsilon)
        """
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # LEAKY RELU
        # Concept: Adds non-linerality
        """
         f(x) = x if x > 0, else 0.1 * x
        """
        self.leakyrelu = nn.LeakyReLU(0.1)

    # Convolution -> Batchnorm -> Activation
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1(nn.Module):
    # in_channels=3 (Red, Green, Blue)
    def __init__(self, in_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels

        # 1. BODY
        # Input: Image(3, 448, 448)
        # Output: Deep Features(1024, 7, 7)
        self.darknet = self._create_conv_layers(self.architecture)

        # 2. HEAD
        # Input: Flattened Features (50176 inputs)
        # Output: The Grid Predictions (1470 outputs)
        self.fully_connected_layers = self._create_fully_connected_layers(**kwargs)

    def forward(self, x):
        # x shape: [Batch, 3, 448, 448]

        # Run the backbone
        x = self.darknet(x)
        # x shape: [Batch, 1024, 7, 7]
        # We have 1024 feature maps, each 7x7 pixels

        # FLATTEN 
        # Concept: The Fully Connected layer cannot understand "3D cubes"
        # It needs a falt list of numbers 
        # Math: 1024 * 7 * 7 = 50176
        return self.fully_connected_layers(torch.flatten(x, start_dim=1))
        # Output shape: [Batch, 1470]

    def _create_conv_layers(self, architecture):
        layers = []
        # STATE TRACKER 
        # We start with 3 channels (RGB)
        # This variable will update as we move through the list:
        # 3 -> 64 -> 192 -> ...
        in_channels = self.in_channels

        for x in architecture:
            # --- CASE 1: STANDARD CONVOLUTION ---
            # Example (7, 64, 2, 3)
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,        # current pipe width (e.g., 3)
                        out_channels=x[1],  # New pipe width (e.g., 64)
                        kernel_size=x[0],   # Window size (e.g., 7x7)
                        stride=x[2],        # step size (e. g., 2)
                        padding=x[3]        # border size (e.g., 3)
                    )
                ]

                in_channels = x[1]

            # --- CASE 2: MAXPOL ---
            elif type(x) == str:
                # Concept: Downsampling
                # We use a 2x2 kernel with Stride 2
                # Math: Divides height and width by 2
                # 448x448 -> 224x224
                # CRITICAL: MaxPool doesn't change the number of channels
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            # --- CASE 3: REPEATING BLOCKS ---
            elif type(x) == list:
                conv1 = x[0] # The "Squeeze" layer (1x1 conv) - tuple
                conv2 = x[1] # The "Expand" layer (3x3 conv) - tuple
                num_repeats = x[2] # how many times to do this - int

                for _ in range(num_repeats):
                    layers += [
                        # STEP 1: SQUEEZE (1x1 Conv)
                        # We compress the depth to save computation
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],  # e.g., 256
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        ),
                        # STEP 2: EXPAND (3x3 conv)
                        # We extract features from the compressed data
                        CNNBlock(
                            in_channels=conv1[1],   # input is the 256 from previous step
                            out_channels=conv2[1],  # output is 512
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fully_connected_layers(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),

            # HIDDEN LAYER 
            # The paper used 4096 nodes
            # We are using 480 in this example to save RAM
            # We compress the 50176 input features into 480 concepts
            nn.Linear(1024 * split_size ** 2, 480), 
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),

            # OUTPUT LAYER 
            # Concept: Map the 480 concepts to the specific Grid Format
            # Math: S * S * (C + B * 5)
            # 7 * 7 (20 + 2 * 5) = 49 * 30 = 1470 nodes
            nn.Linear(480, split_size ** 2 * (num_classes + num_boxes * 5))
        )

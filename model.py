from torch import nn
from utils import *
from math import sqrt
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    """
    VGG base networ to get the  low level features
    """

    def __init__(self):
        super().__init__()

        # Standard VGG network
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceil mode for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 + padding is 1

        # Replacements for FC layers
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # dilated convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained weights
        self.load_pretrained_weights()

    def forward(self, x):
        """
        Forward pass of the network
        :param x: input image
        :return: low level feature maps from conv4_3, conv7
        """
        out = F.relu(self.conv1_1(x))  # (N, 64, H=300, W=300)
        out = F.relu(self.conv1_2(out))  # (N, 64, H=300, W=300)
        out = self.pool1(out)  # (N, 64, H=150, W=150)

        out = F.relu(self.conv2_1(out))  # (N, 128, H=150, W=150)
        out = F.relu(self.conv2_2(out))  # (N, 128, H=150, W=150)
        out = self.pool2(out)  # (N, 128, H=75, W=75)

        out = F.relu(self.conv3_1(out))  # (N, 256, H=75, W=75)
        out = F.relu(self.conv3_2(out))  # (N, 256, H=75, W=75)
        out = F.relu(self.conv3_3(out))  # (N, 256, H=75, W=75)
        out = self.pool3(out)  # (N, 256, H=38, W=38) it is 38 because of ceil_mode=True

        out = F.relu(self.conv4_1(out))  # (N, 512, H=38, W=38)
        out = F.relu(self.conv4_2(out))  # (N, 512, H=38, W=38)
        out = F.relu(self.conv4_3(out))  # (N, 512, H=38, W=38)
        conv4_3_feats = out  # (N, 512, H=38, W=38)  -> this one we use
        out = self.pool4(out)  # (N, 512, H=19, W=19)

        out = F.relu(self.conv5_1(out))  # (N, 512, H=19, W=19)
        out = F.relu(self.conv5_2(out))  # (N, 512, H=19, W=19)
        out = F.relu(self.conv5_3(out))  # (N, 512, H=19, W=19)
        out = self.pool5(out)  # (N, 512, H=19, W=19)

        out = F.relu(self.conv6(out))  # (N, 1024, H=19, W=19)
        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, H=19, W=19)  -> we use this one as well

        return conv4_3_feats, conv7_feats

    def load_pretrained_weights(self):
        """
        Load pretrained weights from VGG16
        the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers
        """
        STATE_DICT = self.state_dict()
        PARAMS = list(STATE_DICT.keys())
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg16_state_dict = vgg16.state_dict()
        PRETRAINED_PARAMS = list(vgg16_state_dict.keys())

        for i, param in enumerate(PARAMS[:-4]):
            STATE_DICT[param] = vgg16_state_dict[PRETRAINED_PARAMS[i]]

        # Convert fc6 and fc7 into conv layers
        conv_fc6_w = vgg16_state_dict['classifier.0.weight'].data.view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_b = vgg16_state_dict['classifier.0.bias'].data  # (4096)
        STATE_DICT['conv6.weight'] = decimate(conv_fc6_w, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        STATE_DICT['conv6.bias'] = decimate(conv_fc6_b, m=[4])  # (1024)

        conv_fc7_w = vgg16_state_dict['classifier.3.weight'].data.view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_b = vgg16_state_dict['classifier.3.bias'].data  # (4096)
        STATE_DICT['conv7.weight'] = decimate(conv_fc7_w, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        STATE_DICT['conv7.bias'] = decimate(conv_fc7_b, m=[4])  # (1024)

        self.load_state_dict(STATE_DICT)
        print('Loaded pretrained weights for VGG16')


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions meant to produce high level feature maps
    """

    def __init__(self):
        super().__init__()

        # Additional convolutions
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # (N, 256, H=19, W=19)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # (N, 512, H=10, W=10)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)  # (N, 128, H=10, W=10)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # (N, 256, H=5, W=5)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)  # (N, 128, H=5, W=5)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)  # (N, 256, H=3, W=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)  # (N, 128, H=3, W=3)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)  # (N, 256, H=1, W=1)

        self.initialize_convolutions()

    def initialize_convolutions(self):
        for conv in self.children():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_uniform_(conv.weight)
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, conv7_feats):
        """
        Forward propagation
        :param conv7_feats: (N, 1024, H=19, W=19)
        :return: high level feature maps (conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
        """

        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, H=19, W=19)
        out = F.relu(self.conv8_2(out))  # (N, 512, H=10, W=10)
        conv8_2_feats = out  # (N, 512, H=10, W=10)

        out = F.relu(self.conv9_1(out))  # (N, 128, H=10, W=10)
        out = F.relu(self.conv9_2(out))  # (N, 256, H=5, W=5)
        conv9_2_feats = out  # (N, 256, H=5, W=5)

        out = F.relu(self.conv10_1(out))  # (N, 128, H=5, W=5)
        out = F.relu(self.conv10_2(out))  # (N, 256, H=3, W=3)
        conv10_2_feats = out  # (N, 256, H=3, W=3)

        out = F.relu(self.conv11_1(out))  # (N, 128, H=3, W=3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, H=1, W=1)

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding box offsets
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of classes
        """
        super().__init__()
        self.n_classes = n_classes

        # Number of prior boxes per feature map location
        self.n_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}

        # Localization (offsets)
        self.loc_conv4_3 = nn.Conv2d(512, self.n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)  # (N, 16, H=38, W=38)
        self.loc_conv7 = nn.Conv2d(1024, self.n_boxes['conv7'] * 4, kernel_size=3, padding=1)  # (N, 24, H=19, W=19)
        self.loc_conv8_2 = nn.Conv2d(512, self.n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)  # (N, 24, H=10, W=10)
        self.loc_conv9_2 = nn.Conv2d(256, self.n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)  # (N, 24, H=5, W=5)
        self.loc_conv10_2 = nn.Conv2d(256, self.n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)  # (N, 16, H=3, W=3)
        self.loc_conv11_2 = nn.Conv2d(256, self.n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)  # (N, 16, H=1, W=1)

        # Class scores
        self.cl_conv4_3 = nn.Conv2d(512, self.n_boxes['conv4_3'] * self.n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, self.n_boxes['conv7'] * self.n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, self.n_boxes['conv8_2'] * self.n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, self.n_boxes['conv9_2'] * self.n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, self.n_boxes['conv10_2'] * self.n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, self.n_boxes['conv11_2'] * self.n_classes, kernel_size=3, padding=1)

        self.initialize_convolutions()

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation
        :param conv4_3_feats: (N, 512, H=38, W=38)
        :param conv7_feats: (N, 1024, H=19, W=19)
        :param conv8_2_feats: (N, 512, H=10, W=10)
        :param conv9_2_feats: (N, 256, H=5, W=5)
        :param conv10_2_feats: (N, 256, H=3, W=3)
        :param conv11_2_feats: (N, 256, H=1, W=1)
        :return: class scores and bounding box offsets for each feature map
        """
        batch_size = conv4_3_feats.size(0)

        # Localization (offsets)
        loc_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, H=38, W=38)
        loc_conv4_3 = loc_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, H=38, W=38, 16)
        # contiguous() returns a neighboring memory block with the same data
        loc_conv4_3 = loc_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4)  -> 5776 = 38*38*4 boxes

        loc_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, H=19, W=19)
        loc_conv7 = loc_conv7.permute(0, 2, 3, 1).contiguous()  # (N, H=19, W=19, 24)
        loc_conv7 = loc_conv7.view(batch_size, -1, 4)  # (N, 2166, 4) -> 2166 = 19*19*6 boxes

        loc_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, H=10, W=10)
        loc_conv8_2 = loc_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, H=10, W=10, 24)
        loc_conv8_2 = loc_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4) -> 600 = 10*10*6 boxes

        loc_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, H=5, W=5)
        loc_conv9_2 = loc_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, H=5, W=5, 24)
        loc_conv9_2 = loc_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4) -> 150 = 5*5*6 boxes

        loc_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, H=3, W=3)
        loc_conv10_2 = loc_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, H=3, W=3, 16)
        loc_conv10_2 = loc_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4) -> 36 = 3*3*4 boxes

        loc_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, H=1, W=1)
        loc_conv11_2 = loc_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, H=1, W=1, 16)
        loc_conv11_2 = loc_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4) -> 4 = 1*1*4 boxes

        # total number of boxes: 5776 + 2166 + 600 + 150 + 36 + 4 = 8732

        # Class scores
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, H=38, W=38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, H=38, W=38, 4 * n_classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes) -> 5776 = 38*38*4 boxes

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, H=19, W=19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, H=19, W=19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes) -> 2166 = 19*19*6 boxes

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, H=10, W=10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, H=10, W=10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes) -> 600 = 10*10*6 boxes

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, H=5, W=5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, H=5, W=5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes) -> 150 = 5*5*6 boxes

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, H=3, W=3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, H=3, W=3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes) -> 36 = 3*3*4 boxes

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, H=1, W=1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, H=1, W=1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes) -> 4 = 1*1*4 boxes

        locs = torch.cat([loc_conv4_3, loc_conv7, loc_conv8_2,
                          loc_conv9_2, loc_conv10_2, loc_conv11_2], dim=1)  # (N, 8732, 4)

        class_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2,
                                  c_conv9_2, c_conv10_2, c_conv11_2], dim=1)  # (N, 8732, n_classes)

        return locs, class_scores

    def initialize_convolutions(self):
        for conv in self.children():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_uniform_(conv.weight)
                nn.init.constant_(conv.bias, 0.0)



class SSD300(nn.Module):
    """
    SSD300 architecture based on the base VGG-16 layers, auxiliary and prediction conv layers.
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes  # number of different types of objects that we want to detect
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since low level features have considerably larger scales, we need to normalize them using L2 norm
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # (1, 512, 1, 1) in conv4_3
        nn.init.constant_(self.rescale_factors, 20)  # initialize with 20

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()  # (8732, 4) -> (cx, cy, w, h)

    def forward(self, image):
        """
        Forward propagation.
        :param image: input images, a tensor of dimensions (N, 3, 300, 300)
        :return: (batch_size, 8732, 4), (batch_size, 8732, n_classes)
        """

        # Run VGG base network to generate lower level features
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Normalize conv4_3_feats using L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        # Run auxiliary network to generate intermediate level features
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 256, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction network to generate localization and class predictions
        locs, class_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats,
                                             conv10_2_feats, conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, class_scores

    @staticmethod
    def create_prior_boxes():
        """
        Create 8732 prior (default) boxes for the 300x300 image size.
        :return: (8732, 4) -> (cx, cy, w, h)
        """
        fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375,
                      'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1.0, 2.0, 0.5], 'conv7': [1.0, 2.0, 3.0, 0.5, 0.333],
                         'conv8_2': [1.0, 2.0, 3.0, 0.5, 0.333], 'conv9_2': [1.0, 2.0, 3.0, 0.5, 0.333],
                         'conv10_2': [1.0, 2.0, 0.5], 'conv11_2': [1.0, 2.0, 0.5]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, we use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map: sqrt(s_k * s_(k+1))

                        if ratio == 1:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.0
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (cx, cy, w, h) -> (cx, cy, w, h) in range [0, 1]

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores and transform them into actual detections.
        Perform Non-Maximum Suppression (NMS) on the resulting detections.
        :param predicted_locs: (N, 8732, 4)
        :param predicted_scores: (N, 8732, n_classes)
        :param min_score: minimum threshold for a detected box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that NMS is not applied to the smaller box
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (N, top_k, 6) -> (N, class, score, x0, y0, x1, y1)
        """

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes, all_images_labels, all_images_scores = [], [], []

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4) fractional

            # Lists to store boxes and scores for this image
            image_boxes, image_labels, image_scores = [], [], []
            max_scores, best_labels = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where the class is the one we're looking for (above min_score)
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score  # (8732)
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores, sort_ind = class_scores[score_above_min_score].sort(dim=0, descending=True)
                class_decoded_locs = decoded_locs[score_above_min_score][sort_ind]  # (n_above_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(
                     class_decoded_locs, class_decoded_locs  # (n_above_min_score, n_above_min_score)
                )

                # Non-Maximum Suppression (NMS)
                # Keep only the boxes that have an IoU of less than 'max_overlap' with the previously selected boxes
                # We'll end up with only the best boxes, as the worst ones will have been removed
                supress = torch.zeros(n_above_min_score, dtype=torch.bool).to(device)  # (n_above_min_score)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box has already been selected for suppression, continue
                    if supress[box] == 1:
                        continue

                    # Suppress boxes whose IoU is greater than 'max_overlap'
                    # Find such boxs and update the suppression vector
                    supress = supress | (overlap[box] > max_overlap).squeeze()

                    # Don't suppress this box, even though it has an IoU of 'max_overlap' with itself
                    supress[box] = 0

                # Store only the best (n_above_min_score - supress.sum()) boxes
                image_boxes.append(class_decoded_locs[~supress])
                image_labels.append(torch.LongTensor((~supress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~supress])

            # If no object of any class was found, add a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0, 0, 1, 1]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0]).to(device))

            # Concatenate the best (n_above_min_score - supress.sum()) boxes for each class found in this image
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top 'k' objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)
                image_scores = image_scores[:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    """
    The MultiBox Loss, as described in the paper.
    This is a combination of:
    (1) the localization loss: Smooth L1 Loss, and
    (2) the classification loss: CrossEntropy Loss
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1):
        """
        :param priors_cxcy: (8732, 4) center-size default boxes in corner-form
        :param threshold: overlap threshold for an anchor to be considered positive
        :param neg_pos_ratio: 3:1 negative:positive ratio
        :param alpha: weighting factor for localization loss
        """
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Compute the loss for both the classification and the localization
        :param predicted_locs: (batch_size, 8732, 4) predicted locations/boxes
        :param predicted_scores: (batch_size, 8732, n_classes) class scores for each of the encoded locations/boxes
        :param boxes: (batch_size, n_objects, 4) object boxes in [xmin, ymin, xmax, ymax] format, there can be a
                      different number of objects in each image in the batch
        :param labels: (batch_size, n_objects) object labels, there can be a different number of objects in each image
                       in the batch
        :return: a scalar tensor containing the loss
        """
        # Compute the loss for classification
        # Find the number of positive and negative examples
        batch_size, n_priors = predicted_locs.size(0), self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (batch_size, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (batch_size, 8732)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # We therefore assign the object to the prior with the highest overlap, and assign the prior to the object.
            _, prior_for_each_object = overlap.max(
                dim=1  # (n_objects) - the prior with the highest overlap for each object
            )

            # We now have a 1:1 mapping between objects and priors, and can assign the priors to the objects.
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure that the priors are not assigned to background, we set the overlap to be greater than 0.5
            overlap_for_each_prior[prior_for_each_object] = 1.

            labels_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlap is < 0.5 to be background (0)
            labels_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            true_classes[i] = labels_for_each_prior  # (8732)

            # Find the regression targets for each prior
            true_locs[i] = cxcy_to_gcxgcy(
                xy_to_cxcy(boxes[i][object_for_each_prior]),  self.priors_cxcy  # (8732, 4)
            )

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (batch_size, 8732)

        # Localization Loss (Smooth L1 Loss) -> only compute loss for positive priors
        loc_loss = self.smooth_l1_loss(
            predicted_locs[positive_priors],  # (n_positive_priors, 4)
            true_locs[positive_priors]  # (n_positive_priors, 4)
        )  # () scalar

        # Classification Loss (Cross Entropy) -> compute loss for the most difficult priors only
        # take the hardest (neg_pos_ration * n_positives) negative priors

        # First, find the number of positive and negative priors for each image in the batch
        n_positives = positive_priors.sum(dim=1)  # (batch_size)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (batch_size)

        # Next, find the loss for all priors
        classification_loss = self.cross_entropy_loss(
            predicted_scores.view(-1, n_classes),  # (batch_size * 8732, n_classes)
            true_classes.view(-1)  # (batch_size * 8732)
        )  # (batch_size * 8732)

        classification_loss = classification_loss.view(batch_size, n_priors)  # (batch_size, 8732)

        # we already know which priors are positive, so we can ignore the classification loss for the negative priors
        classification_loss_pos = classification_loss[positive_priors]  # (sum(n_positives))

        # Next, we find the classification loss for the negative priors
        # To do this, sort ONLY negative priors by their classification loss and take the top n_hard_negatives

        classification_loss_neg = classification_loss.clone()  # (batch_size, 8732)
        classification_loss_neg[positive_priors] = 0.  # (batch_size, 8732)
        classification_loss_neg, _ = classification_loss_neg.sort(dim=1, descending=True)  # (batch_size, 8732)
        # Hardness Ranking -> (batch_size, 8732)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(classification_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (batch_size, 8732)
        classification_loss_hard_neg = classification_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # Finally, compute the total classification loss
        classification_loss = (classification_loss_pos.sum() +
                               classification_loss_hard_neg.sum()) / n_positives.sum().float()  # () scalar

        return classification_loss + self.alpha * loc_loss

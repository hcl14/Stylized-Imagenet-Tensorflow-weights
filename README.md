# Stylized Imagenet Tensorflow weights

Feature weights for VGG16 model for Stylized Imagenet ([https://github.com/rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)), converted from pytorch with feature backpropagation examples. This is my toy test of how Stylized Imagenet features are better for perceptual loss compared to vanilla Imagenet pretrain. Perhaps my implementation is lame and the weights in tensorflow are not reused, you can rewrite it better.

`torch_backprop.py` - testing perceptual loss in pytorch using average pooling, attempting to restore reference image to `out.png`.

`tf_backprop.py` - testing perceptual loss in tensorflow with converted weights, using much better, but more complicated loss, which combines average pooling with max pooling with memorized pooling indices. Attempting to restore reference image to `out1.png`.

To run:

1. You can either clone the author's repository with weights: [https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/src](https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/src)
Then uncomment in `torch_backprop.py` the part which loads weights and generates pickle file.

2. Or just [download generated pickle file with numpy weights for feature layers](https://drive.google.com/file/d/1k9mA4gPedfRSRUdZqRdXi4dLop-ythw_/view?usp=sharing)

Then use `tf_backprop.py` to test feature backpropagation in tensorflow. Other models can be converted this way, if you take time to implement them.

Tested in Tensorflow 1.14, Pytorch 1.1.0, Python 3.5.

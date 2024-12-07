# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for flare removal."""

from flare_removal.python import u_net
from flare_removal.python import vgg
from keras_unet_collection.models import swin_unet_2d, unet_3plus_2d
from keras_unet_collection import base, utils
from flare_removal.python.ViT import ViT, ViTWithDecoder
from flare_removal.python.pretrain_ViT import Pretrain_ViT
from flare_removal.python.u_net_pp import get_model as unet_pp


def build_model(model_type, batch_size, res):
    """Returns a Keras model specified by name."""
    if model_type == 'unet':
        return u_net.get_model(
            input_shape=(res, res, 3),
            scales=4,
            bottleneck_depth=1024,
            bottleneck_layers=2)
      
    elif model_type == 'unet_pp':
        return unet_pp(input_shape=(res, res, 3))
      
    elif model_type == 'unet_3plus_2d':
        return unet_3plus_2d(
            (res, res, 3),
            n_labels=3,
            filter_num_down=[64, 128, 256, 512],
            batch_norm=True,
            pool='max',
            unpool=False,
            #deep_supervision=True,
            weights='imagenet',
            name='unet3plus')

    elif model_type == 'TransUNET':
        return models.transunet_2d((res, res, 3), filter_num=[64, 128, 256, 512], n_labels=3, stack_num_down=2, stack_num_up=2,
                                        embed_dim=512, num_mlp=256, num_heads=6, num_transformer=6,
                                        activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                        batch_norm=True, pool=True, unpool='bilinear', name='transunet')

    elif model_type == 'vit':
      return Pretrain_ViT(
          image_size=224,
          patch_size=16,
          encoder_dim=64,
          decoder_dim=128,
          depth=6,
          heads=4,
          mlp_dim=128,
          dropout=0.1,
          decoder_depth=4
      )

    elif model_type == 'can':
        return vgg.build_can(
            input_shape=(512, 512, 3), conv_channels=64, out_channels=3)
    else:
        raise ValueError(model_type)

import tensorflow as tf
from .u_net import _down_block, _up_block


def _nested_block(x, skips, depth, stage, interpolation='bilinear', name_prefix='nested'):
    """Handles the dense skip connections for U-Net++.

    Args:
        x: The input tensor for the current stage.
        skips: List of skip connections from previous downscaling levels.
        depth: Number of output channels.
        stage: Current depth level.
        interpolation: Interpolation method for upsampling.
        name_prefix: Layer name prefix.

    Returns:
        A tensor that represents the fused output of this nested block.
    """
    outputs = [x]
    for i, skip in enumerate(skips):
        # Fuse the input tensor and skip connection at this level.
        upsampled = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation=interpolation,
            name=f'{name_prefix}_upsample_stage{stage}_level{i}'
        )(outputs[-1])
        merged = tf.keras.layers.concatenate(
            [upsampled, skip], name=f'{name_prefix}_concat_stage{stage}_level{i}'
        )
        conv = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{name_prefix}_conv_stage{stage}_level{i}'
        )(merged)
        outputs.append(conv)
    return outputs[-1]  # Return the most refined output for this stage.


def get_model(input_shape=(512, 512, 3), scales=4, bottleneck_depth=1024, bottleneck_layers=2):
    """Builds a U-Net++ with dense skip connections.

    Args:
        input_shape: Shape of the input tensor without batch dimension.
        scales: Number of downscaling/upscaling blocks.
        bottleneck_depth: Number of channels in the bottleneck.
        bottleneck_layers: Number of Conv2D layers in the bottleneck.

    Returns:
        A Keras model instance representing a U-Net++.
    """
    input_layer = tf.keras.Input(shape=input_shape, name='input')
    previous_output = input_layer

    # Downscaling arm with skip connections.
    skips = []
    depths = [bottleneck_depth // 2**i for i in range(scales, 0, -1)]
    for depth in depths:
        skip, previous_output = _down_block(
            previous_output, depth, name_prefix=f'down{depth}'
        )
        skips.append(skip)

    # Bottleneck.
    for i in range(bottleneck_layers):
        previous_output = tf.keras.layers.Conv2D(
            filters=bottleneck_depth,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'bottleneck_conv{i + 1}'
        )(previous_output)

    # Upscaling arm with nested dense skip connections.
    nested_skips = [[] for _ in range(len(skips))]
    for i, (depth, skip) in enumerate(zip(reversed(depths), reversed(skips))):
        nested_skips[i].append(skip)
        previous_output = _nested_block(
            previous_output, nested_skips[i], depth,
            stage=i, name_prefix=f'nested_up{depth}'
        )
        nested_skips[i].append(previous_output)

    # Squash output to (0, 1).
    output_layer = tf.keras.layers.Conv2D(
        filters=input_shape[-1],
        kernel_size=1,
        activation='sigmoid',
        name='output'
    )(previous_output)

    return tf.keras.Model(input_layer, output_layer, name='unet_pp')


# Helper functions (_down_block and _up_block) remain unchanged.

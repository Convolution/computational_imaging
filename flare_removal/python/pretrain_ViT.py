from transformers import ViTFeatureExtractor, TFAutoModel, ViTConfig
import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import Layer
from einops import rearrange, repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.net = Sequential([
            nn.Dense(units=hidden_dim, activation='gelu'),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)
        self.to_out = Sequential([
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        x = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.to_out(x, training=training)

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = [PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)) for _ in range(depth)]
        self.mlp_layers = [PreNorm(MLP(dim, mlp_dim, dropout=dropout)) for _ in range(depth)]

    def call(self, x, training=True):
        for attn, mlp in zip(self.layers, self.mlp_layers):
            x = attn(x, training=training) + x
            x = mlp(x, training=training) + x
        return x

class Pretrain_ViT(tf.keras.Model):
    def __init__(self, image_size, patch_size, encoder_dim, decoder_dim, depth, heads, mlp_dim, dim_head=64, 
                 dropout=0.0, decoder_depth=4):
        super(Pretrain_ViT, self).__init__()

        # Image dimensions
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * 3  # RGB channels

        # Load pretrained encoder (from Hugging Face)
        self.encoder = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k", from_pt=True)

        self.encoder_projection = tf.keras.layers.Dense(units=decoder_dim, name="encoder_projection")

        # Decoder positional embedding
        self.decoder_pos_embedding = tf.Variable(tf.random.normal([1, num_patches, decoder_dim]))

       # Decoder Transformer
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=heads, dim_head=dim_head,
                                   mlp_dim=decoder_dim * 4, dropout=dropout)

        # Reconstruction head for output
        self.reconstruction_head = Sequential([
            tf.keras.layers.Dense(units=patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                      h=image_height // patch_height, p1=patch_height, p2=patch_width)
        ])

    def call(self, img, training=True):
        # Preprocess input image for the pretrained encoder
        # shape = tf.shape(img)
        # if shape[1] == 224 and shape[2] == 224 and shape[3] == 3:
        img = tf.transpose(img, perm=[0, 3, 1, 2])
        encoder_features = self.encoder(img, training=training)["last_hidden_state"]
        # Remove [CLS] token
        encoder_features = encoder_features[:, 1:, :]  # Shape: [1, 196, 64]
        
        encoder_features = self.encoder_projection(encoder_features)
       
        # Add positional embeddings (match number of patches)
        x = encoder_features + self.decoder_pos_embedding  # Shape: [1, 196, 128]
       
        # Pass through decoder
        x = self.decoder(x, training=training)

        # Reconstruct patches and return
        reconstructed = self.reconstruction_head(x)
        return reconstructed


# #Instantiate the model
# model =Pretrain_ViT(
#     image_size=224,
#     patch_size=16,
#     encoder_dim=64,
#     decoder_dim=128,
#     depth=6,
#     heads=4,
#     mlp_dim=128,
#     dropout=0.1,
#     decoder_depth=4
# )

# # Use ViTFeatureExtractor to preprocess the input
# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# # Fix input image range
# raw_img = tf.random.uniform([1, 224, 224, 3], minval=0, maxval=255, dtype=tf.float32)
# raw_img_uint8 = tf.cast(raw_img, tf.uint8)  # Cast to uint8 for compatibility with feature extractor

# # Preprocess the image
# processed_img = feature_extractor(images=raw_img_uint8.numpy(), return_tensors="tf")["pixel_values"]

# # Forward pass
# output = model(processed_img)
# print("Input shape:", processed_img.shape)
# print("Output shape:", output.shape)

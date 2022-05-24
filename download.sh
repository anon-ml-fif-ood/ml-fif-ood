source .env.example
source .env

# BiT-S-R101
wget -P $PRE_TRAINED_DIR https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz 
# ViT-L-16
wget -P $PRE_TRAINED_DIR https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz

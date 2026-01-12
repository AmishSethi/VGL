"""
SongUNet implementation specifically for count conditioning.
"""

from unet_models_song import SongUNet


def SongUNet_Count_XL(**kwargs):
    return SongUNet(
        conditioning_type='radius',  # Count uses same conditioning as radius (scalar value)
        model_channels=192,
        channel_mult=[1,2,3,4],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Count_L(**kwargs):
    return SongUNet(
        conditioning_type='radius',
        model_channels=192,
        channel_mult=[1,2,2,2],
        num_blocks=3,
        **kwargs
    )

def SongUNet_Count_B(**kwargs):
    return SongUNet(
        conditioning_type='radius',
        model_channels=128,
        channel_mult=[1,2,2,2],
        num_blocks=4,
        **kwargs
    )

def SongUNet_Count_S(**kwargs):
    return SongUNet(
        conditioning_type='radius',
        model_channels=128,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )


# Model name mappings
SongUNet_Count_models = {
    'SongUNet-Count-XL': SongUNet_Count_XL,
    'SongUNet-Count-L': SongUNet_Count_L,
    'SongUNet-Count-B': SongUNet_Count_B,
    'SongUNet-Count-S': SongUNet_Count_S,
}

# Alias for compatibility
SongUNet_models = SongUNet_Count_models


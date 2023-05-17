try:
    from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
except:
    LPIPSWithDiscriminator = None
from ldm.modules.losses.vae_loss import VAELoss

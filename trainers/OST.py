import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from einops import rearrange

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'OST',
                      "vision_depth": cfg.TRAINER.OST.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.OST.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.OST.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.OST.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def returnCLIP(config, logger=None,
               class_names=None, des_spatial=None, des_temporal=None, class_emb=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building OST")
    model = OST(config, des_spatial, des_temporal, class_emb, clip_model, logger)

    if config.TRAINER.OST.PROMPT_MODEL:
        logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.OST.USE
        if train_complete_clip == "both":
            logger.info("Turning on gradients for COMPLETE OST model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        else:
            if train_complete_clip == "image":
                logger.info("Turning on gradients for image side the OST model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            else:
                logger.info("Turning on gradients for TEXT side the OST model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model


class OST(nn.Module):
    def __init__(self, cfg, des_spatial, des_temporal, class_emb, clip_model, logger):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_encoder = TextEncoder(clip_model)
        self.des_spatial_token = rearrange(des_spatial, '(a n)d -> a n d', n=cfg.TRAINER.OST.N)
        self.des_temporal_token = rearrange(des_temporal, '(a n)d -> a n d', n=cfg.TRAINER.OST.N)
        self.class_emb_token = class_emb
        self.N = cfg.TRAINER.OST.N
        self.max_iter = 100    
        self.eps = 0.1
        
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u) # Initialize r as a tensor of ones with the same shape as u
        c = torch.ones_like(v) # Initialize c as a tensor of ones with the same shape as v
        thresh = 1e-2 # Threshold to determine convergence in Sinkhorn iterations
        # Sinkhorn iteration
        for i in range(self.max_iter): # Iterate up to the maximum number of iterations
            r0 = r # Save the previous iteration's r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1) # Update r
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1) # Update c
            err = (r - r0).abs().mean() # Calculate the mean absolute change in iterations
            if err.item() < thresh: # If the change is below the threshold, stop iterating
                break
        P = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K # Obtain the final transport
        return P    # transport plan P 
        
    def OptimalDescriptorSolver(self, image_emb, descriptor_emb, logit_scale):
        A, N, D = descriptor_emb.shape # Get the shape of descriptor embeddings
        B, T, _ = image_emb.shape # Get the shape of video embeddings
        sim = torch.einsum('b t d, a n d->t n b a', image_emb, descriptor_emb).contiguous() # Compute the similarity
        sim = rearrange(sim, 't n b a -> (b a)t n') # Rearrange dimensions
        cost_mat = 1 - sim # Calculate the cost matrix
        pp_x = torch.zeros(B*A, T, dtype=sim.dtype, device=sim.device).fill_(1. / T) # Initialize the horizontal probability vector
        pp_y = torch.zeros(B*A, N, dtype=sim.dtype, device=sim.device).fill_(1. / N) # Initialize the vertical probability vector
        with torch.no_grad():
            KK = torch.exp( - cost_mat / self.eps) # Calculate the cost matrix with exponentiation
            P = self.Sinkhorn(KK, pp_x,pp_y) # Apply Sinkhorn algorithm to obtain the optimal transport plan P
        if torch.isnan(P).any():
            return None
        # using transport plan P to obtain logits
        score_ot = torch.sum(P * sim, dim=(1, 2)) # Frobenius inner product
        logits = score_ot.contiguous().view(B, A) * logit_scale # Classification logits
        return logits    
    
    
    def get_logit(self, image_features, descriptor_spatial, descriptor_temporal, logit_scale):
        logits_temporal = self.OptimalDescriptorSolver(image_features, descriptor_temporal, logit_scale)
        logits_spatial = self.OptimalDescriptorSolver(image_features, descriptor_spatial, logit_scale)     
        logits_spatial_pool = logit_scale * image_features.mean(dim=1, keepdim=False) @ descriptor_spatial.mean(dim=1, keepdim=False).t() 
        logits_temporal_pool = logit_scale * image_features.mean(dim=1, keepdim=False) @ descriptor_temporal.mean(dim=1, keepdim=False).t() 
        logit_spatial = logits_spatial_pool + logits_spatial
        logit_temporal = logits_temporal_pool + logits_temporal
        logits = 0.5 * logit_temporal + 0.5 * logit_spatial
        return logits
    
    def inference(self, image):
        logit_scale = self.logit_scale.exp()
        # b = image.shape[0]
        # Lets encode the video into required format
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image.type(self.dtype))
        # Now again attach the batch dimensions
        image_features = image_features.view(b, t, -1)  # [B, T, 512]
        # Now take the mean along the temporal direction

        # Finally, make the text features
        
        descriptor_spatial = self.text_encoder(rearrange(self.des_spatial_token, 'a n d -> (a n) d'))
        descriptor_temporal = self.text_encoder(rearrange(self.des_temporal_token, 'a n d -> (a n) d'))
        descriptor_spatial = rearrange(descriptor_spatial, '(a n) d -> a n d', n=self.N)
        descriptor_temporal = rearrange(descriptor_temporal, '(a n) d -> a n d', n=self.N)
        
        descriptor_spatial = descriptor_spatial / descriptor_spatial.norm(dim=-1, keepdim=True)
        descriptor_temporal = descriptor_temporal / descriptor_temporal.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits_temporal = self.OptimalDescriptorSolver(image_features, descriptor_temporal, logit_scale)
        logits_spatial = self.OptimalDescriptorSolver(image_features, descriptor_spatial, logit_scale)     
        logits_spatial_pool = logit_scale * image_features.mean(dim=1, keepdim=False) @ descriptor_spatial.mean(dim=1, keepdim=False).t() 
        logits_temporal_pool = logit_scale * image_features.mean(dim=1, keepdim=False) @ descriptor_temporal.mean(dim=1, keepdim=False).t() 
        
        logit_spatial = logits_spatial_pool + logits_spatial
        logit_temporal = logits_temporal_pool + logits_temporal
        logits = 0.5 * logit_temporal + 0.5 * logit_spatial
        return logits
        
    def forward(self, image, label_id):
        logit_scale = self.logit_scale.exp()
        # b = image.shape[0]
        # Lets encode the video into required format
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image.type(self.dtype))
        # Now again attach the batch dimensions
        image_features = image_features.view(b, t, -1)  # [B, T, 512]

        local_spatial_token = self.des_spatial_token[label_id]
        local_temporal_token = self.des_temporal_token[label_id] # 4, 4, d
        
        descriptor_spatial = self.text_encoder(rearrange(local_spatial_token, 'a n d -> (a n) d'))
        descriptor_temporal = self.text_encoder(rearrange(local_temporal_token, 'a n d -> (a n) d'))
        descriptor_spatial = rearrange(descriptor_spatial, '(a n) d -> a n d', n=self.N)
        descriptor_temporal = rearrange(descriptor_temporal, '(a n) d -> a n d', n=self.N)
        
        descriptor_spatial = descriptor_spatial / descriptor_spatial.norm(dim=-1, keepdim=True)
        descriptor_temporal = descriptor_temporal / descriptor_temporal.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, descriptor_spatial, descriptor_temporal, logit_scale
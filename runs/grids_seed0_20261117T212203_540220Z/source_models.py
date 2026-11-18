"""Compact proposal-based implementations, not an exact reproduction of the paper."""
import torch
from torch import nn
import torch.nn.functional as F
from .data import CAMERAS

VARIANTS = ('vision_only','dense','uniform','grids','grids_film')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),
                                    nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                                    nn.Conv2d(32,64,3,2,1),nn.ReLU())
    def forward(self,x): return self.layers(x)

def regular_grid(size, device=None):
    # Pixel-center coordinates for align_corners=False. Coordinates are (x,y).
    axis = (torch.arange(size,device=device)+0.5)*2/size-1
    y,x = torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack((x,y),dim=-1).reshape(-1,2)

class Policy(nn.Module):
    def __init__(self, variant, vocab_size):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(variant)
        self.variant = variant
        self.encoder = Encoder()
        self.words = nn.Embedding(vocab_size,32,padding_idx=0)
        # Order-sensitive instruction encoder, trained from scratch.
        self.language = nn.GRU(32,32,batch_first=True)
        self.coordinate_embedding = nn.Linear(2,64)
        self.token_projection = nn.Sequential(nn.Linear(64,64),nn.ReLU())
        self.camera_embedding = nn.Embedding(2,64)
        # Same attention aggregation and action head across all five variants.
        self.query = nn.Linear(32,64)
        self.key = nn.Linear(64,64)
        self.value = nn.Linear(64,64)
        self.head = nn.Sequential(nn.Linear(96,256),nn.ReLU(),nn.Linear(256,7))
        self.register_buffer('uniform_coordinates',regular_grid(4))
        if variant in ('grids','grids_film'):
            self.coordinate_predictor = nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,32))
            nn.init.normal_(self.coordinate_predictor[-1].weight,std=0.001)
            nn.init.zeros_(self.coordinate_predictor[-1].bias)
        if variant == 'grids_film':
            self.film = nn.Linear(32,128)
            nn.init.normal_(self.film.weight,std=0.001)
            nn.init.zeros_(self.film.bias)

    @property
    def tokens_per_camera(self):
        return 256 if self.variant in ('vision_only','dense') else 16

    def language_features(self,text):
        if self.variant == 'vision_only':
            return torch.zeros(text.shape[0],32,device=text.device)
        encoded,_ = self.language(self.words(text))
        lengths = text.ne(0).sum(1).clamp_min(1)
        return encoded[torch.arange(len(text),device=text.device), lengths-1]

    def forward(self, images, text, return_coordinates=False):
        language = self.language_features(text)
        tokens, positions = [], {}
        for camera_index,camera in enumerate(CAMERAS):
            features = self.encoder(images[camera])
            batch,channels,height,width = features.shape
            if self.variant in ('vision_only','dense'):
                coords = regular_grid(height,features.device).unsqueeze(0).expand(batch,-1,-1)
                selected = features.flatten(2).transpose(1,2)
            else:
                coords = self.uniform_coordinates.unsqueeze(0).expand(batch,-1,-1)
                if self.variant in ('grids','grids_film'):
                    pooled = features.mean(dim=(2,3))
                    if self.variant == 'grids_film':
                        scale,shift = self.film(language).chunk(2,dim=-1)
                        pooled = pooled*(1+scale)+shift
                    offset = self.coordinate_predictor(pooled).reshape(batch,16,2)
                    coords = torch.tanh(torch.atanh(coords)+offset)
                sampled = F.grid_sample(features,coords.unsqueeze(2),mode='bilinear',
                                        padding_mode='border',align_corners=False)
                selected = sampled.squeeze(-1).transpose(1,2)
            positions[camera] = coords
            # Coordinate injection preserves the spatial meaning of selected features.
            selected = selected + self.coordinate_embedding(coords)
            selected = selected + self.camera_embedding.weight[camera_index]
            tokens.append(self.token_projection(selected))
        tokens = torch.cat(tokens,dim=1)
        scores = (self.key(tokens)*self.query(language).unsqueeze(1)).sum(-1)/(64**0.5)
        attention = scores.softmax(dim=1)
        pooled = (self.value(tokens)*attention.unsqueeze(-1)).sum(1)
        actions = self.head(torch.cat((pooled,language),dim=-1))
        return (actions,positions) if return_coordinates else actions

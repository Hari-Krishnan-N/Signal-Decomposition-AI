import torch 
from torch import nn


class ImageEncoder(nn.Module):

    def __init__(self, channels=8, embedding_dim=1024, nheads=8, dim_fc=256, num_layers=4, dropout=0.1):
        
        super(ImageEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.patch_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channels)
        )
        self.dropout = nn.Dropout(p=dropout)

        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, activation='relu', dim_feedforward=dim_fc, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

    def forward(self, x, key_mask):
        x = self.dropout(self.patch_conv(x))
        x = x.view(x.shape[0], -1, self.embedding_dim)
        x = self.encoder(x, src_key_padding_mask=key_mask)
        return x
    


class AudioEncoder(nn.Module):

    def __init__(self, embedding_dim=1024, nheads=8, dim_fc=256, num_layers=4, dropout=0.1):
        super(AudioEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, activation='relu', dim_feedforward=dim_fc, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

    def forward(self, x, key_mask):
        return self.encoder(x, src_key_padding_mask=key_mask)
    


class CoattentionModel(nn.Module):

    def __init__(self, channels=8, embedding_dim=1024, nheads=4, dim_fc=256, num_layers=6, dropout=0.1):

        super(CoattentionModel, self).__init__()

        self.img_encoder = ImageEncoder(channels=channels, embedding_dim=embedding_dim, nheads=nheads, dim_fc=dim_fc, num_layers=4, dropout=dropout)
        self.audio_encoder = AudioEncoder(embedding_dim=embedding_dim, nheads=nheads, dim_fc=dim_fc, num_layers=4, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, activation='relu', dim_feedforward=dim_fc, batch_first=True, dropout=dropout)
        self.final_encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, image, audio, img_mask, audio_mask):
        image = self.dropout(self.img_encoder(image, key_mask=img_mask))
        audio = self.dropout(self.audio_encoder(audio, key_mask=audio_mask))
        x = torch.cat([image, audio], dim=1)
        final_mask = torch.cat([img_mask, audio_mask], dim=1)
        x = self.final_encoder(x, src_key_padding_mask=final_mask)
        x = x.mean(dim=1)
        #x = image.mean(dim=1)
        #x = audio.mean(dim=1)
        return self.final_fc(x)

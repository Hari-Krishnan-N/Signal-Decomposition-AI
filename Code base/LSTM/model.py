import torch 
from torch import nn


class ImageEncoder(nn.Module):

    def __init__(self, channels=10, embedding_dim=1024, num_layers=4, dropout=0.1):
        
        super(ImageEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.patch_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channels)
        )
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=int(embedding_dim/4), num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True, proj_size=int(embedding_dim/8))

    def forward(self, x):
        x = self.dropout(self.patch_conv(x))
        x = x.view(x.shape[0], -1, self.embedding_dim)
        x = self.encoder(x)
        return x
    


class AudioEncoder(nn.Module):

    def __init__(self, embedding_dim=1024, num_layers=4, dropout=0.1):
        super(AudioEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=int(embedding_dim/4), num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True, proj_size=int(embedding_dim/8))

    def forward(self, x):
        return self.encoder(x)
    


class LSTMClassifier(nn.Module):

    def __init__(self, channels=10, embedding_dim=1024, num_layers=4, dropout=0.1, mode:str=None):

        super(LSTMClassifier, self).__init__()

        self.mode = mode
        self.img_encoder = ImageEncoder(channels=channels, num_layers=num_layers)
        self.audio_encoder = AudioEncoder(num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)

        self.final_encoder = nn.LSTM(input_size=int(embedding_dim/4), hidden_size=int(embedding_dim/4), num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True, proj_size=int(embedding_dim/8))
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=int(embedding_dim/4), out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, image, audio):

        if self.mode == 'image':
            image = self.dropout(self.img_encoder(image)[0])
            x = image.mean(dim=1)
        elif self.mode == 'audio':
            audio = self.dropout(self.audio_encoder(audio)[0])
            x = audio.mean(dim=1)
        else:
            image = self.img_encoder(image)
            audio = self.audio_encoder(audio)
            x = torch.cat([self.dropout(image[0]), self.dropout(audio[0])], dim=1)
            h = image[1][0] + audio[1][0]
            c = image[1][1] + audio[1][1]
            x = self.dropout(self.final_encoder(x, (h,c))[0])
            x = x.mean(dim=1)
        return self.final_fc(x)

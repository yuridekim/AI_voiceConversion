import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_GLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv2d_GLU, self).__init__()
        inC = kwargs.get("inC", 0)
        outC = kwargs.get("outC", 0)
        k = kwargs.get("k", 0)
        s = kwargs.get("s", 0)
        p = kwargs.get("p", 0)
        T = kwargs.get("transpose", False)

        if T:
            self.cnn = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
        else:
            self.cnn = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)

        self.cnn_norm = nn.BatchNorm2d(outC)
        self.gate_norm = nn.BatchNorm2d(outC)
        
    def forward(self, x):
        
        h1 = self.cnn_norm(self.cnn(x))
        h2 = self.gate_norm(self.gate(x))
        out = torch.mul(h1, torch.sigmoid(h2))
        return out

def attach_style(inputs, style):
    style = style.view(style.size(0), style.size(1), 1, 1)
    style = style.repeat(1, 1, inputs.size(2), inputs.size(3))
    inputs_bias_added = torch.cat([inputs, style], dim=1)
    return inputs_bias_added

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.style_dim = kwargs.get("style_dim", 4)
        self.latent_dim = kwargs.get("latent_dim", 8)
        self.vae_type = kwargs.get("vae_type", '')

        assert self.vae_type in ['VAE1', 'VAE2', 'VAE3', 'MD'], "VAE type error"

        """
        (1, 36, 128) => (5, 36, 128) => (10, 18, 64) => (10, 9, 32) => (16, 1, 32)
        (k-s)/2 = p
        """
        C_structure = [8, 32, 32, self.latent_dim]
        k_structure = [(3,9), (4,8), (4,8), (9,5)]
        s_structure = [(1,1), (2,2), (2,2), (9,1)]

        layer_num = len(C_structure)

        inC = 1
        self.convs= nn.ModuleList([])
        for layer_idx in range(layer_num):
            if self.vae_type in ['VAE3', 'MD']:
                inC += self.style_dim
            outC = C_structure[layer_idx]
            k = k_structure[layer_idx]
            s = s_structure[layer_idx]
            p = ((k[0]-s[0])//2, (k[1]-s[1])//2)

            if layer_idx == layer_num-1:
                self.conv_mu = nn.Conv2d(inC, outC, k, s, padding=p)
                self.conv_logvar = nn.Conv2d(inC, outC, k, s, padding=p)
            else:
                self.convs.append(
                    nn.Sequential(
                        Conv2d_GLU(inC=inC, outC=outC, k=k, s=s, p=p),
                        nn.Dropout2d(0.3)
                    )
                )
                inC = outC
        
    def forward(self, x, style):
        h = x
        if self.vae_type in ['VAE3', 'MD']:
            h = attach_style(h, style)
        for conv in self.convs:
            h = conv(h)
            if self.vae_type in ['VAE3', 'MD']:
                h = attach_style(h, style)

        h_mu = self.conv_mu(h)
        h_logvar = self.conv_logvar(h)
        o = reparameterize(h_mu, h_logvar)
     
        return h_mu, h_logvar, o


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.style_dim = kwargs.get("style_dim", 4)
        self.latent_dim = kwargs.get("latent_dim", 8)
        self.vae_type = kwargs.get("vae_type", '')

        assert self.vae_type in ['VAE3', 'MD'], "VAE type error"

        """
        (1, 36, 128) => (5, 36, 128) => (10, 18, 64) => (10, 9, 32) => (16, 1, 32)
        (k-s)/2 = p
        """
        C_structure = [32, 32, 8, 1]
        k_structure = [(9,5), (4,8), (4,8), (3,9)]
        s_structure = [(9,1), (2,2), (2,2), (1,1)]

        layer_num = len(C_structure)

        inC = self.latent_dim
        self.convs= nn.ModuleList([])
        if self.vae_type in ['VAE3']:
            inC += self.style_dim
        for layer_idx in range(layer_num):
            outC = C_structure[layer_idx]
            k = k_structure[layer_idx]
            s = s_structure[layer_idx]
            p = ((k[0]-s[0])//2, (k[1]-s[1])//2)

            if layer_idx == layer_num-1:
                self.conv_mu = nn.ConvTranspose2d(inC, outC, k, s, padding=p)
                self.conv_logvar = nn.ConvTranspose2d(inC, outC, k, s, padding=p)
            else:
                self.convs.append(
                    nn.Sequential(
                        Conv2d_GLU(inC=inC, outC=outC, k=k, s=s, p=p, transpose=True),
                        nn.Dropout2d(0.3)
                    )
                )
            inC = outC
            if self.vae_type in ['VAE3']:
                inC += self.style_dim
        
    def forward(self, x, style):
        h = x
        if self.vae_type in ['VAE3']:
            h = attach_style(h, style)
    
        for conv in self.convs:
            
            h = conv(h)
            if self.vae_type in ['VAE3']:
                h = attach_style(h, style)

        h_mu = self.conv_mu(h)
        h_logvar = self.conv_logvar(h)
        o = reparameterize(h_mu, h_logvar)
     
        return h_mu, h_logvar, o
class LatentClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        latent: (8, 1, 32) => (16, 1, 16) => (32, 1, 8) => (16, 1, 4) => (spk_dim, 1, 1)
        """
        super(LatentClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.label_num = kwargs.get("label_num", 4)

        self.conv1 = Conv2d_GLU(inC=self.latent_dim, outC=16, k=(1,4), s=(1,2), p=(0,1))
        self.conv2 = Conv2d_GLU(inC=16, outC=32, k=(1,4), s=(1,2), p=(0,1))
        self.conv3 = Conv2d_GLU(inC=32, outC=16, k=(1,4), s=(1,2), p=(0,1))

        self.conv_out = nn.Conv2d(16, self.label_num, (1, 4),(1, 4), padding=(0, 0))
        self.label_out = nn.LogSoftmax(dim=1)

    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.conv2(h1)
        h = self.conv3(h2)

        o = self.conv_out(h)
        o = o.view(-1, self.label_num)
        o = self.label_out(o)

        return o

class DataClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        data: (1, 36, 128) => (8, 18, 64) => (16, 9, 32) => (32, 8, 16) => (16, 4, 8) => (label_dim, 1, 1)
        """
        super(DataClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_num = kwargs.get("label_num", 4)

        self.conv1 = Conv2d_GLU(inC=1, outC=8, k=(4,4), s=(2,2), p=(1,1))
        self.conv2 = Conv2d_GLU(inC=8, outC=16, k=(4,4), s=(2,2), p=(1,1))
        self.conv3 = Conv2d_GLU(inC=16, outC=32, k=(4,4), s=(2,2), p=(1,1))
        self.conv4 = Conv2d_GLU(inC=32, outC=16, k=(4,4), s=(2,2), p=(1,1))

        self.conv_out = nn.Conv2d(16, self.label_num, (2, 8),(2, 8), padding=(0, 0))
        self.label_out = nn.LogSoftmax(dim=1)

    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h = self.conv4(h3)

        o = self.conv_out(h)
        o = o.view(-1, self.label_num)
        o = self.label_out(o)

        return o



import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation_patch import PatchKey, Patcher
from core_attention import InferenceCore, GenerationCore

# model for attention
# it contains only attention model ( not pyramid, tower and pool)
class GQNAttention(nn.Module):
    def __init__(self, L=12, shared_core=False):
        super(GQNAttention, self).__init__()

        # Number of generative layers
        self.L = L

        # Patch layers
        self.patcher = Patcher()
        self.patchKey = PatchKey()

        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])

        self.eta_pi = nn.Conv2d(64, 2*3, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(64, 2*3, kernel_size=5, stride=1, padding=2)

    # EstimateELBO
    #### parameters
    #### x : context images
    #### v : poses
    #### v_q : queried pose
    #### x_q : ground truth image
    #### sigma
    #### variables
    #### B : batch size
    #### M : num of context images
    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()

        # Scene encoder
        patch_r= self.patcher(x,v)
        key_images = self.patchKey(x)

        # Generator initial state
        c_g = x.new_zeros((B, 64, 8, 8))
        h_g = x.new_zeros((B, 64, 8, 8))
        u = x.new_zeros((B, 64, 32, 32))

        # Inference initial state
        c_e = x.new_zeros((B, 64, 8, 8))
        h_e = x.new_zeros((B, 64, 8, 8))

        elbo = 0
        for l in range(self.L):

            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # attention
            r = torch.Tensor().cuda()
            for i in range(B):
                attn_weight = torch.sum(key_images[i] * h_g[i],1).reshape(-1,1,64)
                attn_weight = F.softmax(attn_weight,-1).reshape(-1,1,8,8).repeat(1,64,1,1)
                attn_feature = (torch.sum(patch_r[i] * attn_weight,0)).unsqueeze(0)
                r = torch.cat((r,attn_feature),0)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

            # ELBO KL contribution update
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        # ELBO likelihood contribution update
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[1,2,3])

        return elbo

    def generate(self, x, v, v_q):
        B, M, *_ = x.size()

        # Scene encoder
        patch_r= self.patcher(x,v)
        key_images = self.patchKey(x)

        # Generator initial state
        c_g = x.new_zeros((B, 64, 8, 8))
        h_g = x.new_zeros((B, 64, 8, 8))
        u = x.new_zeros((B, 64, 32, 32))

        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # attention
            r = torch.Tensor().cuda()
            for i in range(B):
                attn_weight = torch.sum(key_images[i] * h_g[i],1).reshape(-1,1,64)
                attn_weight = F.softmax(attn_weight,-1).reshape(-1,1,8,8).repeat(1,64,1,1)
                attn_feature = (torch.sum(patch_r[i] * attn_weight,0)).unsqueeze(0)
                r = torch.cat((r,attn_feature),0)

            # Prior sample
            z = pi.sample()

            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

        # Image sample
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)

    def kl_divergence(self, x, v, v_q, x_q):    # for MULTI-gpu
        B, M, *_ = x.size()

        # Scene encoder
        patch_r= self.patcher(x,v)
        key_images = self.patchKey(x)

        # Generator initial state
        c_g = x.new_zeros((B, 64, 8, 8))
        h_g = x.new_zeros((B, 64, 8, 8))
        u = x.new_zeros((B, 64, 32, 32))

        # Inference initial state
        c_e = x.new_zeros((B, 64, 8, 8))
        h_e = x.new_zeros((B, 64, 8, 8))

        kl = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # attention
            r = torch.Tensor().cuda()
            for i in range(B):
                attn_weight = torch.sum(key_images[i] * h_g[i],1).reshape(-1,1,64)
                attn_weight = F.softmax(attn_weight,-1).reshape(-1,1,8,8).repeat(1,64,1,1)
                attn_feature = (torch.sum(patch_r[i] * attn_weight,0)).unsqueeze(0)
                r = torch.cat((r,attn_feature),0)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        return kl

    def reconstruct(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        patch_r= self.patcher(x,v)
        key_images = self.patchKey(x)

        # Generator initial state
        c_g = x.new_zeros((B, 64, 8, 8))
        h_g = x.new_zeros((B, 64, 8, 8))
        u = x.new_zeros((B, 64, 32, 32))

        # Inference initial state
        c_e = x.new_zeros((B, 64, 8, 8))
        h_e = x.new_zeros((B, 64, 8, 8))

        for l in range(self.L):

            # attention
            r = torch.Tensor().cuda()
            for i in range(B):
                attn_weight = torch.sum(key_images[i] * h_g[i],1).reshape(-1,1,64)
                attn_weight = F.softmax(attn_weight,-1).reshape(-1,1,8,8).repeat(1,64,1,1)
                attn_feature = (torch.sum(patch_r[i] * attn_weight,0)).unsqueeze(0)
                r = torch.cat((r,attn_feature),0)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)

        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)

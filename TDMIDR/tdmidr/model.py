#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np
import torch.nn as nn
tl.set_backend('pytorch')

def initialize_A_transpose_ridge(W, p, lambda_var, lambda_12):
    T, n = W.shape
    
    # W_future (T-p) x n
    W_future = W[p:]  # 从 p+1 到 T
    
    #  W_past (T-p) x (n*p)
    W_past = np.zeros((T - p, n * p))
    for t in range(p, T):
        W_past[t - p] = np.concatenate([W[t - i - 1] for i in range(1, p + 1)])  # 拼接 W_{t-1}, ..., W_{t-p}
    
    #
    I = np.eye(n * p)  

    A_transpose = np.linalg.solve(W_past.T @ W_past + (lambda_12 / lambda_var) * I, W_past.T @ W_future)
    
    A_list = [A_transpose[i * n : (i + 1) * n, :].T for i in range(p)]  # [A_1, A_2, ..., A_p]

 
    return A_list


class TDMIDR(nn.Module):
    def __init__(self, input_shape, ranks, lambda_var=1e-4, lambda_smooth=1e-4, 
                 lambda_graph=1e-4, lambda_L2=1e-4,lambda_L1=1e-4, p=2, device='cpu', 
                  X=None, mask=None, L1=None, L2=None, use_hosvd=True): # 图拉普拉斯矩阵
        super(TDMIDR, self).__init__()
        self.device = device
        self.shape = input_shape
        self.ranks = ranks
        self.lambda_var = lambda_var
        self.lambda_smooth = lambda_smooth
        self.lambda_graph = lambda_graph
        self.lambda_L2 = lambda_L2
        self.lambda_L1 = lambda_L1
        self.p = p
        self.L1 = L1.to(device) if L1 is not None else None
        self.L2 = L2.to(device) if L2 is not None else None

        I, J, K = input_shape
        R1, R2, R3 = ranks

        
        if use_hosvd:
            U_init, V_init, W_init, G_init = self.hosvd_initialize(X, ranks, mask)

            self.A_list = nn.ParameterList([
                nn.Parameter(torch.tensor(A_i, dtype=torch.float32, device=device), requires_grad=True) 
                for A_i in initialize_A_transpose_ridge(W_init.detach().cpu().numpy(), p, lambda_var, lambda_L2)
            ])
        else:
            U_init = torch.randn(I, R1, device=device) * 0.1
            V_init = torch.randn(J, R2, device=device)* 0.1
            W_init = torch.randn(K, R3, device=device)* 0.1
            G_init = torch.randn(R1, R2, R3, device=device)* 0.1
            self.A_list = nn.ParameterList([
                            nn.Parameter(torch.randn(R3, R3, device=device)* 0.1, requires_grad=True)
                            for _ in range(p)
                        ])

        
        T = W_init.shape[0]  
        D2_np = (np.eye(T, k=-1) - 2 * np.eye(T, k=0) + np.eye(T, k=1))[1:-1]
        self.D2 = torch.tensor(D2_np, dtype=torch.float32).to(device)
        
        
        self.U = nn.Parameter(U_init)
        self.V = nn.Parameter(V_init)
        self.W = nn.Parameter(W_init)
        self.G = nn.Parameter(G_init)
        self.mask = mask

    def hosvd_initialize(self, X, ranks, mask):
        """HOSVD 初始化 U, V, W, G"""
        tensor = tl.tensor(X)

        mask_np = mask.float()

        core, factors = tucker(tensor, rank=ranks, mask=mask_np, init='svd', tol=1e-6, random_state=36)
 
        U_init = factors[0].clone().detach().to(self.device).float()
        V_init = factors[1].clone().detach().to(self.device).float()
        W_init = factors[2].clone().detach().to(self.device).float()
        G_init = core.clone().detach().to(self.device).float()

        return U_init, V_init, W_init, G_init

    def forward(self):

        # TX_hat = G ×₁ U ×₂ V ×₃ W
        X1 = torch.einsum('abc,ia->ibc', self.G, self.U)
        X2 = torch.einsum('ibc,jb->ijc', X1, self.V)
        X_hat = torch.einsum('ijc,kc->ijk', X2, self.W)
        

        return X_hat, self.G, self.U, self.V, self.W, self.A_list

    def loss(self, X_obs, mask):
        X_hat, G, U, V, W, A_list = self.forward()
        
       
        mse_loss = ((mask * (X_hat - X_obs))**2).sum() / mask.sum()

        var_loss = 0.0
        for t in range(self.p, W.shape[0]):
          
            pred = sum([W[t - i - 1] @ A_list[i].T for i in range(self.p)])  
            var_loss += torch.norm(W[t] - pred)**2
        var_loss *= self.lambda_var / 2

        
        smooth_diff = self.D2 @ W  # (T-2, R3)
        smooth_loss = torch.norm(smooth_diff, p='fro')**2 * self.lambda_smooth / 2

       
        graph_loss = 0.0
        if self.L1 is not None:
            graph_loss += torch.trace(U.T @ self.L1 @ U)
        if self.L2 is not None:
            graph_loss += torch.trace(V.T @ self.L2 @ V)
        graph_loss *= self.lambda_graph

        
        l2_reg = (torch.norm(U)**2 + torch.norm(V)**2 + 
                 torch.norm(W)**2 + torch.norm(G)**2 + 
                 sum(torch.norm(A_i.T)**2 for A_i in self.A_list))
        l2_reg *= self.lambda_L2 / 2

        
        l1_reg = self.lambda_L1 * (torch.norm(V, p=1) + torch.norm(G, p=1))

        total_loss = mse_loss + var_loss + smooth_loss + graph_loss + l2_reg + l1_reg

        
        return total_loss

    def project_parameters(self):
        
        with torch.no_grad():
#
            Q, R = torch.linalg.qr(self.V, mode='reduced')
            self.V.data = Q
            self.G.data = torch.einsum('abc,jb->ajc', self.G, R)
            
    def set_requires_grad(self, group):
       
        groups = {
            'G': [self.G],
            'U': [self.U],
            'V': [self.V],
            'W': [self.W],
            'A': self.A_list 
        }

       
        for param in self.parameters():
            param.requires_grad_(False)


        def _unfreeze(params):
            for p in params:
                if isinstance(p, (list, tuple)):
                    _unfreeze(p)  
                elif isinstance(p, nn.Parameter):
                    p.requires_grad_(True)
                else:
                    raise ValueError(f"Unsupported parameter type: {type(p)}")

        
        target_params = groups.get(group, [])
        _unfreeze(target_params)
        
 
        def _flatten(params):
           
            flat_list = []
            for p in params:
                if isinstance(p, (list, tuple)):
                    flat_list.extend(_flatten(p))
                elif isinstance(p, nn.Parameter):
                    flat_list.append(p)
            return flat_list

        flat_params = _flatten(target_params)
        for p in flat_params:
            if not p.requires_grad:
                raise RuntimeError(f"Parameter {p} was not properly unfrozen!")


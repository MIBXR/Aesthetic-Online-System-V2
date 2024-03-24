import torch.nn as nn
import torch
from models.mv2 import cat_net


def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = cat_net()
        self.base_model = base_model
        #设置权重
        for p in self.parameters():
            p.requires_grad = False

#         self.l1 = torch.nn.Linear(3681, 100)
        self.head1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 1),
        )
        self.head2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 1),
        )
#         self.head3 = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(p=0.75),
#             nn.Linear(3681, 1),
#         )
#         self.head4 = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(p=0.75),
#             nn.Linear(3681, 1),
#         )
#         self.head5 = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(p=0.75),
#             nn.Linear(3681, 1),
#         )
#         self.head6 = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(p=0.75),
#             nn.Linear(3681, 1),
#         )

    def forward(self, x):

        x1, x2 = self.base_model(x)
        x = torch.cat([x1,x2],1)
        
#         x = self.l1(x)
        X1 = self.head1(x)
        X2 = self.head2(x)
#         X3 = self.head3(x)
#         X4 = self.head4(x)
#         X5 = self.head5(x)
#         X6 = self.head6(x)
        
#         return torch.stack([X1, X2, X3, X4, X5, X6],dim=1)
        X = torch.cat([X1, X2],1)
        return X

    def get_last_shared_layer(self):
        return self.base_model
#
# if __name__=='__main__':
#     model = NIMA()
#     x = torch.rand((16,3,224,224))
#     out = model(x)
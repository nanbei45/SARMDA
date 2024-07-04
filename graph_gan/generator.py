from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv
import torch.nn.functional as F
from torchvision import models


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, hidden_channels)


    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.conv5(x,edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False,heads=1,dropout=0.6)
        #self.lin1 = Linear(hidden_channels, hidden_channels)

        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False,heads=1,dropout=0.6)
        #self.lin2 = Linear(hidden_channels, hidden_channels)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False, heads=1,dropout=0.6)
        #self.lin3 = Linear(hidden_channels, hidden_channels)

    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        return x

class Model(torch.nn.Module):
    def __init__(self,hidden_channels,data):
        super().__init__()
        self.micro_lin = torch.nn.Linear(1052,hidden_channels)
        self.micro_emb = torch.nn.Embedding(data["micro"].num_nodes,hidden_channels)
        self.disease_lin = torch.nn.Linear(218, hidden_channels)
        self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels=hidden_channels)
        self.gnn = to_hetero(self.gnn,metadata=data.metadata()) #转换成异构图

    def encode(self,data:HeteroData)->Tensor:
        #先拿到字典
        x_dict={
            "micro": self.micro_lin(data["micro"].x)+self.micro_emb(data["micro"].node_id),
            "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
        }

        x_dict = self.gnn(x_dict,data.edge_index_dict)
        return x_dict

    def decode(self,z, edge_label_index):
        return (z['micro'][edge_label_index[0]] * z['disease'][edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z['micro'] @ z['disease'].t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 64),
        )

    def forward(self,inputs):
        return self.model(inputs)



class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)
            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
        self.init_weights()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
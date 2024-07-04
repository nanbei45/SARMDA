import torch
from torch import Tensor, nn
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, SAGEConv, to_hetero, HANConv, HeteroConv, GCNConv, GATv2Conv, GraphConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=1)
        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=1)
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels,hidden_channels)

        self.conv4 = SAGEConv(hidden_channels, hidden_channels)

        self.conv5 = SAGEConv(hidden_channels, hidden_channels)


    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.conv5(x,edge_index)
        return x
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
    def forward(self, x:Tensor, edge_index:Tensor)->Tensor:
        x = F.relu(self.conv1(x,edge_index))
        x = self.conv2(x,edge_index)
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
        return prob_adj,(prob_adj > 0).nonzero(as_tuple=False).t()

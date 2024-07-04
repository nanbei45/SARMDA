import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.utils import negative_sampling

from model import Model
from torch_geometric.data import HeteroData
import torch.nn.functional as F


micro_dis = pd.read_csv("mirco-dis2.csv", header=None)
micro_dis.columns =['micro_id','dis_id']
def make_edge_index(network):
    source_nodes = micro_dis['micro_id'].values
    target_nodes = micro_dis['dis_id'].values
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]),dtype=torch.long)
    return edge_index

def make_edge_attr(m_sim,d_sim):
    micro_dis = pd.read_csv("mirco-dis2.csv", header=None)
    micro_dis = micro_dis.values.tolist()
    edge_attr = np.zeros((len(micro_dis), len(micro_dis)), dtype=float)
    i=0
    for list in micro_dis:
        micro,dis = list
        for j in range(len(m_sim[micro])):
            edge_attr[i][j] = m_sim[micro][j]
        for p in range(len(d_sim[dis])):
            edge_attr[i][len(m_sim)+p] = d_sim[dis][p]
        i += 1
    return edge_attr


association = pd.read_csv("./data2/index_association.csv", header=0, index_col=0).values
m_sim = pd.read_csv("./data2/microbe_features.txt", header=None, index_col=None ,sep='\t').values
d_sim = pd.read_csv("./data2/disease_features.txt", header=None, index_col=None ,sep='\t').values


data = HeteroData()
edge_index = make_edge_index(association)
edge_attr = make_edge_attr(m_sim,d_sim)
print(edge_attr.shape)
data['micro'].node_id = torch.arange(len(m_sim))
data['disease'].node_id = torch.arange(len(d_sim))
data['micro'].x = torch.tensor(m_sim,dtype=torch.float32)
data['disease'].x = torch.tensor(d_sim,dtype=torch.float32)
data['micro','cause','disease'].edge_index = edge_index
data['micro','cause','disease'].edge_arrt = edge_attr
data = ToUndirected()(data)

# 按照一定比例分割数据集为训练集、测试集和验证集
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3, #监督边占训练集的比例，不是必须的，可以为0
    neg_sampling_ratio = 1.0, #负采用比例，二分类任务，有一个正样本和负样本，大部分是负样本
    add_negative_train_samples=False, #训练不增加负样本，每一次迭代选择固定的负样本
    # 告诉PyG边的连接关系
    edge_types=[('micro', 'cause', 'disease')],
    rev_edge_types=[('disease', 'rev_cause', 'micro')],

    )
# 分割数据集
train_data, val_data, test_data = transform(data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = data.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
val_data = val_data.to(device)


model = Model(hidden_channels=16,data=data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

# 生成正负样本边的标记
def get_link_labels(train_data, neg_edge_index):
    num_links = train_data['micro', 'cause', 'disease'].edge_label_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)  # 向量
    link_labels[:train_data['micro', 'cause', 'disease'].edge_label_index.size(1)] = 1
    return link_labels


def train():
    model.train()
    optimizer.zero_grad()
    #节点表征学习
    z = model.encode(train_data)
    #从训练集中采样与正边相同数量的负边，每个epoch的负采样样本都不同，随机更具有健壮性
    neg_edge_label_index = negative_sampling(edge_index=train_data['micro', 'cause', 'disease'].edge_index,
                                           num_nodes=([train_data['micro'].num_nodes,train_data['disease'].num_nodes]),
                                             num_neg_samples=train_data['micro', 'cause', 'disease'].edge_label_index.size(1),
                                             method='sparse')
    edge_label_index = torch.cat(
        [train_data['micro', 'cause', 'disease'].edge_label_index, neg_edge_label_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data['micro', 'cause', 'disease'].edge_label,
        train_data['micro', 'cause', 'disease'].edge_label.new_zeros(neg_edge_label_index.size(1))
    ], dim=0)
    #计算有无边的概率
    out = model.decode(z,edge_label_index).view(-1)

    #真实边的情况
    loss = F.binary_cross_entropy_with_logits(out,edge_label)
    loss.backward()
    optimizer.step()
    return loss

def test(data):
    model.eval()
    z = model.encode(data)

    out = model.decode(z, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())
def acc_test(data):
    model.eval()
    z = model.encode(data)

    out = model.decode(z, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    out = out.cpu().detach().numpy()
    for i in range(out.shape[0]):
        if out[i]>= 0.5:
            out[i] = 1
        else:
            out[i] = 0
    out = out.astype(int)
    return accuracy_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out)

def f1_test(data):
    model.eval()
    z = model.encode(data)

    out = model.decode(z, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    out = np.around(out.cpu().detach().numpy(), 0).astype(int)
    return f1_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out)

def all_test(data):
    model.eval()
    z = model.encode(data)

    out = model.decode(z, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    auc = roc_auc_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())
    out = np.around(out.cpu().detach().numpy(), 0).astype(int)
    recall = recall_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(),out)
    precision = precision_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(),out)
    f1 = f1_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out)
    acc = accuracy_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out)
    return f1_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out)

# with open("sim/result/data1/data2_baseline","w") as csvfile:
#     writer = csv.writer(csvfile)
#     csv_head = ["Test","val"]
#     writer.writerow(csv_head)

best_val_auc = final_test_auc = 0
for epoch in range(0,200):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    #
    # with open("sim/result/data1/data2_baseline", "a+") as csvfile:
    #     writer = csv.writer(csvfile)
    #     row_data = [test_auc,val_auc]
    #     writer.writerow(row_data)


    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        torch.save(model.state_dict(), "result/data2/data2_GCN_model.pt")

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

print(f'Final Test:{final_test_auc:.4f}')

z = model.encode(test_data)
scroce,final_edge_index = model.decode_all(z)
# np.savetxt('final_index_data2.txt',final_edge_index.cpu().detach().numpy())
# np.savetxt('final_data2.txt',scroce.cpu().detach().numpy())
print(final_edge_index.shape)
print(scroce.shape)
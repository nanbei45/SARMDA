import csv
import random

import numpy as np
import pandas as pd
import torch
import torchcontrib
from torchcontrib.optim import SWA
from sklearn.metrics import roc_auc_score
from torch import optim, nn
from torch.autograd import Variable
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F


from graph_gan.discriminator import Discriminator, MLP_D
from graph_gan.generator import Generator, Model, MLP_G, ModelEma

'''
Wgan-Gp 判别器损失：损失为正值或者负值均可，表示的真数据和假数据的相对分布位置，收敛到0为目标
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_gradient_penalty(D, real, fake):
    # 每一个样本对应一个sigma。样本个数为64，特征数为512：[64,512]
    sigma = torch.rand(real.size(0), 1)  # [1052,1]
    sigma = sigma.expand(real.size())  # [1052, 64]

    # 按公式计算x_hat
    sigma = sigma.to(device)
    x_hat = sigma*real +(torch.tensor(1.)-sigma)*fake

    #x_hat.requires_grad = True
    # 为得到梯度先计算y
    d_x_hat = D(x_hat)
    grad_outputs = torch.ones(d_x_hat.size())
    grad_outputs = grad_outputs.to(device)


    # 计算梯度,autograd.grad返回的是一个元组(梯度值，)
    gradients = torch.autograd.grad(outputs=d_x_hat, inputs=x_hat.to(device),
                                    grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def normal_pdf(x, mu, sigma):
    '''正态分布的概率密度函数'''
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf



micro_dis = pd.read_csv("../mirco-dis2.csv", header=None)
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



association = pd.read_csv("../data2/index_association.csv", header=0, index_col=0).values
m_sim = pd.read_csv("../data2/microbe_features.txt", header=None, index_col=None ,sep='\t').values
d_sim = pd.read_csv("../data2/disease_features.txt", header=None, index_col=None ,sep='\t').values



data = HeteroData()
edge_index = make_edge_index(association)

data['micro'].node_id = torch.arange(len(m_sim))
data['disease'].node_id = torch.arange(len(d_sim))
data['micro'].x = torch.tensor(m_sim,dtype=torch.float32)
data['disease'].x = torch.tensor(d_sim,dtype=torch.float32)
data['micro','cause','disease'].edge_index = edge_index



data = ToUndirected()(data)
print(data)

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
data = data.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
val_data = val_data.to(device)




def np_decode(z, edge_label_index):
    return (z[edge_label_index[0]] * z[edge_label_index[1]+1052]).sum(dim=-1)

def np_test(fake_data,data):

    out = np_decode(fake_data, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())

def train(model,optimizer,train_data,cosine_schedule):
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
    cosine_schedule.step()
    return loss,edge_label_index ,edge_label

def train_gan_g(G_model,D_model,optimizer_g,edge_label_index):
    """
    Training WGAN generator network
    :return: error of generator part
    """
    G_model.train()
    G_model.zero_grad()

    noise = Variable(torch.ones(len(edge_label_index[0]), 32)).to(device)
    noise.data.normal_(0, 1)
    one = torch.FloatTensor([1])

    fake_hidden = G_model(noise)
    errG = D_model(fake_hidden)
    #out = D_model(fake_hidden)
    #errG = F.binary_cross_entropy_with_logits(out,torch.tensor(1.).to(device))

    # loss / backprop
    errG.backward()
    optimizer_g.step()

    return errG


def get_link_emb(train_data,z,hidden):
    edge_label_index = train_data["micro","cause","disease"].edge_label_index
    real_hidden = torch.zeros((len(edge_label_index[0]),hidden))
    for idx in range(len(edge_label_index[0])):
        x = z["micro"][edge_label_index[0][idx]].view(1,16)
        y = z["disease"][edge_label_index[1][idx]].view(1,16)
        real_hidden[idx] = torch.cat((x, y), 1)
    return real_hidden

def train_gan_d(model,D_model,G_model,train_data,optimizer_d,optimizer,edge_label_index,edge_label,cosine_schedule):
    # WGAN Weight clipping
    for p in D_model.parameters():
        p.data.clamp_(-0.01, 0.01)

    D_model.train()
    D_model.zero_grad()
    # positive samples ----------------------------
    real_hidden_ = model.encode(train_data)
    real_hidden = get_link_emb(train_data,real_hidden_,32)
    real_hidden = real_hidden.to(device)
    # negative samples ----------------------------
    noise = Variable(torch.ones(len(edge_label_index[0]), 32)).to(device)
    noise.data.normal_(0, 1)
    fake_hidden = G_model(noise)
    errD_real = D_model(real_hidden)
    errD_fake = D_model(fake_hidden.detach())
    gradient_penalty = cal_gradient_penalty(D_model, real_hidden, fake_hidden)
    errD_fake += 0.1*gradient_penalty
    errD_real += 0.1*gradient_penalty
    errD_real.backward(retain_graph=True)
    errD_fake.backward(retain_graph=True)
    errD = -(errD_real - errD_fake) +0.1*gradient_penalty
    optimizer_d.step()
    # This is the version of Wasserstein GAN, which has gradient clipping
    torch.nn.utils.clip_grad_norm_(D_model.parameters(),0.01)
    return errD, errD_real, errD_fake



def neg_sampling(train_data):
    # 从训练集中采样与正边相同数量的负边，每个epoch的负采样样本都不同，随机更具有健壮性
    neg_edge_label_index = negative_sampling(edge_index=train_data['micro', 'cause', 'disease'].edge_index,
                                             num_nodes=(
                                             [train_data['micro'].num_nodes, train_data['disease'].num_nodes]),
                                             num_neg_samples=train_data[
                                                 'micro', 'cause', 'disease'].edge_label_index.size(1),
                                             method='sparse')
    edge_label_index = torch.cat(
        [train_data['micro', 'cause', 'disease'].edge_label_index, neg_edge_label_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data['micro', 'cause', 'disease'].edge_label,
        train_data['micro', 'cause', 'disease'].edge_label.new_zeros(neg_edge_label_index.size(1))
    ], dim=0)
    return edge_label_index,edge_label

def newnewtrain(model,optimizer):
    G = MLP_G(ninput=64, noutput=1270, layers='300-300')
    D = MLP_D(ninput=1270, noutput=1, layers='300-300')

    d_learning_rate = 0.001
    g_learning_rate = 0.001
    optimizer_D = optim.Adam(D.parameters(), lr=d_learning_rate)
    optimizer_G = optim.Adam(G.parameters(), lr=g_learning_rate)
    D.to(device)
    G.to(device)
    best_val_auc = final_test_auc = 0
    # crossEntropy loss for discriminator
    criterion_ce = nn.CrossEntropyLoss()

    model.train()
    #gan作为一个正则项
    for epoch in range(0,100):
        D.train()
        G.train()
        noise = torch.randn(1270, 64).to(device)
        fake = G(noise) #从噪音中产生离真实数据分布尽可能接近的数据

        z = model.encode(train_data)
        edge_label_index, edge_label = neg_sampling(train_data)
        out = model.decode(z, edge_label_index).view(-1)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)

        real = torch.cat((z["micro"], z["disease"]), dim=0)  #真实样本即为上面的encoder中的编码器Encoder所产生的低维向量
        '''所以当我们对GAN进行训练时，同时也将正负样本之间的差异反馈给Encoder，从而使得1）Encoder需要提取更有效的代表节点的信息以区分伪样本；2）Encoder需要避免过拟合，以免太容易被生成器学习。'''
        d_real = D(real)
        d_fake = D(fake)
        gradient_penalty = cal_gradient_penalty(D, real, fake)
        loss_d = -(torch.mean(d_real) - torch.mean(d_fake)) + 0.5 * gradient_penalty
        optimizer_D.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizer_D.step()

        noise = torch.rand(1270,64).to(device)
        fake = G(noise)
        gen_fake = D(fake)
        loss_g = -torch.mean(gen_fake)
        optimizer_G.zero_grad()
        loss_g.backward(retain_graph=True)
        optimizer_G.step()

        loss_m = 0.5 * loss + 0.5 * loss_g
        optimizer.zero_grad()
        loss_m.backward()
        optimizer.step()

        model.eval()
        G.eval()
        with torch.no_grad():

            fake_data = G(noise)
            train_auc = np_test(fake_data, train_data)
            val_auc = np_test(fake_data,val_data)
            test_auc = np_test(fake_data,test_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            print(f'Epoch: {epoch:03d}, Loss_d: {loss_d:.4f},Train: {train_auc:.4f}, Val: {val_auc:.4f}, '
                  f'Test: {test_auc:.4f}')

        print(f'Final Test:{final_test_auc:.4f}')


def test(model,data):
    model.eval()
    z = model.encode(data)
    out = model.decode(z, data["micro", "cause", "disease"].edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data["micro", "cause", "disease"].edge_label.cpu().detach().numpy(), out.cpu().detach().numpy())



# with open("../logs/gan/gan_auc3.csv","w") as csvfile:
#     writer = csv.writer(csvfile)
#     csv_head = ["Test","Val"]
#     writer.writerow(csv_head)

def mytest():
    G = MLP_G(ninput=32, noutput=32, layers='32-32')
    D = MLP_D(ninput=32, noutput=1, layers='32-32')
    model = Model(hidden_channels=16, data=data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=0.0004)

    model.to(device)

    d_learning_rate = 0.001
    g_learning_rate = 0.001
    optimizer_d = optim.Adam(D.parameters(), lr=d_learning_rate)
    optimizer_g = optim.Adam(G.parameters(), lr=g_learning_rate)
    D.to(device)
    G.to(device)
    best_val_auc = final_test_auc = 0
    # crossEntropy loss for discriminator
    criterion_ce = nn.CrossEntropyLoss().to(device)
    niter_gan = 1
    niters_gan_schedule = "2-4-6-10-20-30-40"
    gan_schedule = [int(x) for x in niters_gan_schedule.split("-")]

    for epoch in range(100):
        if epoch in gan_schedule:
            niter_gan += 1

        model.train()
        optimizer.zero_grad()
        # 节点表征学习
        z = model.encode(train_data)

        # 从训练集中采样与正边相同数量的负边，每个epoch的负采样样本都不同，随机更具有健壮性
        neg_edge_label_index = negative_sampling(edge_index=train_data['micro', 'cause', 'disease'].edge_index,
                                                 num_nodes=(
                                                 [train_data['micro'].num_nodes, train_data['disease'].num_nodes]),
                                                 num_neg_samples=train_data[
                                                     'micro', 'cause', 'disease'].edge_label_index.size(1),
                                                 method='sparse')
        edge_label_index = torch.cat(
            [train_data['micro', 'cause', 'disease'].edge_label_index, neg_edge_label_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data['micro', 'cause', 'disease'].edge_label,
            train_data['micro', 'cause', 'disease'].edge_label.new_zeros(neg_edge_label_index.size(1))
        ], dim=0)
        # 计算有无边的概率
        out = model.decode(z, edge_label_index).view(-1)
        # 真实边的情况
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        edge_label_index = train_data["micro", "cause", "disease"].edge_label_index
        edge_label = train_data["micro", "cause", "disease"].edge_label

        for k in range(1):
            for i in range(5):
                errD, errD_real, errD_fake = \
                    train_gan_d(model, D, G, train_data, optimizer_d, optimizer, edge_label_index,edge_label,cosine_schedule)

            for i in range(1):
                errG = train_gan_g(G, D, optimizer_g,edge_label_index)

        loss = loss+0.1*errD
        loss.backward()
        optimizer.step()
        cosine_schedule.step()

        print('[%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
              'Loss_D_fake: %.8f) Loss_G: %.8f Loss: %.8f'
              % (epoch, 100,
                 errD, errD_real,
                 errD_fake, errG,loss))

        model.eval()

        with torch.no_grad():
            #ModelEma(model=model,decay=0.9999,device=device)

            val_auc = test(model, val_data)
            test_auc = test(model, test_data)
            # with open("../logs/gan/gan_auc3.csv", "a+") as csvfile:
            #     writer = csv.writer(csvfile)
            #     row_data = [test_auc,val_auc]
            #     writer.writerow(row_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
                torch.save(model.state_dict(), "../result/data2/data2_best_model.pt")
            print(f'Epoch: {epoch:03d}, Val: {val_auc:.4f}, '
                  f'Test: {test_auc:.4f}')

        print(f'Final Test:{final_test_auc:.4f}')



if __name__ == '__main__':

    mytest()



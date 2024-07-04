import csv
import pandas as pd
from py2neo import Node,Relationship,Graph,NodeMatcher,RelationshipMatcher
import numpy as np

from qianduan.utils import readcsv
import torch


def dis_micro_neo4j():
    micro, dis = readcsv()
    print(micro)
    np.savetxt('../qianduan/data/micro.txt', micro, delimiter='\n', fmt='%s')
    np.savetxt('../qianduan/data/disease.txt', dis, delimiter='\n', fmt='%s')
    micro_dis = np.loadtxt("../final.txt")

    graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
    graph.delete_all()
    matcher = NodeMatcher(graph)
    for i in range(len(micro_dis[0])):
        a = matcher.match("Microorganism", name=micro[int(micro_dis[0][i])]).exists()
        if a == False:
            node1 = Node("Microorganism", name=micro[int(micro_dis[0][i])])
            graph.create(node1)
        else:
            node1 = matcher.match("Microorganism", name=micro[int(micro_dis[0][i])]).first()

        b = matcher.match("Disease", name=dis[int(micro_dis[1][i])]).exists()
        if b == False:
            node2 = Node("Disease", name=dis[int(micro_dis[1][i])])
            graph.create(node2)
        else:
            node2 = matcher.match("Disease", name=dis[int(micro_dis[1][i])]).first()
        print(node1, node2)
        relation1 = Relationship(node1, "cause", node2)
        relation2 = Relationship(node2, "caused", node1)
        graph.create(relation1)
        graph.create(relation2)


def micro2_micro1():
    micro1, _ = readcsv()
    micro2 = pd.read_excel("../qianduan/data/aBiofilm/microbes.xlsx").values
    micro_dict = {}
    for i in range(len(micro2)):
        micro2_i = micro2[i][0]
        micro2Split = micro2_i.split(" ")
        micro_dict[micro2_i] = []
        for j in range(len(micro1)):
            micro1_j = micro1[j]
            micro1Split = micro1_j.split(" ")
            if micro1Split[0] == micro2Split[0]:
                micro_dict[micro2_i].append(micro1_j)
    return micro_dict

def drug_micro_neo4j():
    micro2 = pd.read_excel("../qianduan/data/aBiofilm/microbes.xlsx").values
    micro2_micro1_dict = micro2_micro1()
    drug = pd.read_excel("../qianduan/data/aBiofilm/drugs.xlsx").values
    drug_micro = pd.read_excel("../qianduan/data/aBiofilm/associations.xlsx").values
    graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
    matcher = NodeMatcher(graph)

    for i in range(len(drug_micro)):
        drug_i_index = drug_micro[i][0] -1
        micro_i_index = drug_micro[i][1] -1
        drug_i = drug[drug_i_index][0]
        micro_i = micro2[micro_i_index][0]
        drug_node_exits = matcher.match("Drug", name=drug[drug_i_index][0]).exists()
        if drug_node_exits==False:
            drug_node = Node("Drug", name=drug[drug_i_index][0])
            graph.create(drug_node)
        else:
            drug_node = matcher.match("Drug", name=drug[drug_i_index][0]).first()

        for j in range(len(micro2_micro1_dict[micro2[micro_i_index][0]])):
            micro1_j_exits = matcher.match("Microorganism", name=micro2_micro1_dict[micro2[micro_i_index][0]][j]).exists()
            if micro1_j_exits == False:
                micro_j = Node("Microorganism", name=micro2_micro1_dict[micro2[micro_i_index][0]][j])
                graph.create(micro_j)
            else:
                micro1_j = matcher.match("Microorganism", name=micro2_micro1_dict[micro2[micro_i_index][0]][j]).first()
            relation_j = Relationship(micro1_j, "treated", drug_node)
            graph.create(relation_j)
            print(relation_j)


def make_data1_neo4j():
    micro, dis = readcsv("../data/index_association.csv")
    micro_dis_index = np.loadtxt("../result/data1/data1_tuple.txt")
    micro_dis_score = np.loadtxt("../result/data1/data1_score.txt")
    print(micro_dis_index.shape)
    print(micro_dis_score.shape)
    association = pd.read_csv("../data/index_association.csv", header=0, index_col=0).values
    print(association.shape)

    graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
    #graph.delete_all()
    matcher = NodeMatcher(graph)
    for i in range(len(micro_dis_index[0])):
        micro_index = int(micro_dis_index[0][i])
        disease_index = int(micro_dis_index[1][i])
        micro_node_exist = matcher.match("Microorganism", name=micro[micro_index]).exists()
        if micro_node_exist == False:
            micro_node = Node("Microorganism", name=micro[micro_index])
            graph.create(micro_node)
        else:
            micro_node = matcher.match("Microorganism", name=micro[micro_index]).first()

        disease_node_exist = matcher.match("Disease", name=dis[disease_index]).exists()
        if disease_node_exist == False:
            disease_node = Node("Disease", name=dis[disease_index])
            graph.create(disease_node)
        else:
            disease_node = matcher.match("Disease", name=dis[disease_index]).first()

        score = micro_dis_score[micro_index][disease_index]
        properties = {"sim_score": score,"source":"HMDAD"}
        if association[disease_index][micro_index] == 1:

            relation = Relationship(disease_node, "already_exits", micro_node, **properties)

        else:
            relation = Relationship(disease_node, "new_find", micro_node, **properties)

        graph.create(relation)



if __name__ == "__main__":
    micro, dis = readcsv("../data2/index_association.csv")
    micro_dis_index = np.loadtxt("../result/data2/data2_tuple.txt")
    micro_dis_score = np.loadtxt("../result/data2/data2_score.txt")
    micro_dis_score = torch.sigmoid(torch.tensor(micro_dis_score))
    micro_dis_score = np.array(micro_dis_score)
    print(micro_dis_index.shape)
    print(micro_dis_score)
    association = pd.read_csv("../data2/index_association.csv", header=0, index_col=0).values
    print(association.shape)

    graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
    graph.delete_all()
    matcher = NodeMatcher(graph)
    relation_matcher = RelationshipMatcher(graph)
    for i in range(len(micro_dis_index[0])):
        micro_index = int(micro_dis_index[0][i])
        disease_index = int(micro_dis_index[1][i])
        micro_node_exist = matcher.match("Microorganism", name=micro[micro_index]).exists()
        if micro_node_exist == False:
            micro_node = Node("Microorganism", name=micro[micro_index])
            graph.create(micro_node)
        else:
            micro_node = matcher.match("Microorganism", name=micro[micro_index]).first()

        disease_node_exist = matcher.match("Disease", name=dis[disease_index]).exists()
        if disease_node_exist == False:
            disease_node = Node("Disease", name=dis[disease_index])
            graph.create(disease_node)
        else:
            disease_node = matcher.match("Disease", name=dis[disease_index]).first()

        score = micro_dis_score[micro_index][disease_index]

        relation_exist = relation_matcher.match({disease_node,micro_node},None).exists()
        if relation_exist == False:
            if association[disease_index][micro_index] == 1:
                properties = {"sim_score": score, "source": "Disbiome","label":"already_exists"}
                relation = Relationship(disease_node, "already_exists", micro_node,**properties)
            else:
                properties = {"sim_score": score, "source": "Disbiome", "label": "new_find"}
                relation = Relationship(disease_node, "new_find", micro_node,**properties)
            graph.create(relation)
        else:
            relation = relation_matcher.match({disease_node, micro_node}, None).first()
            relation["source"] = "HMDAD、Disbiome"




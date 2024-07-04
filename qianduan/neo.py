import pandas as pd
import streamlit as st
from streamlit_chatbox import *
from streamlit_agraph import agraph, Config
from match_mov_peo import match_things
from st_aggrid import AgGrid, GridOptionsBuilder
from py2neo import Graph, Node, Relationship, NodeMatcher, Subgraph,RelationshipMatcher

from qianduan.testnoe4j import readcsv

# graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
# node_matcher = NodeMatcher(graph)
# relationship_matcher = RelationshipMatcher(graph)
# centerNode = node_matcher.match("Disease").where(name="Acne").first()
# relationship = list(relationship_matcher.match((centerNode, None), r_type="recause"))
# all_nodes = []
# for i in range(len(relationship)):
#     all_nodes.append(relationship[i].end_node["name"])
#
# answer = "与该疾病相关的微生物有："
# for i in range(len(all_nodes)):
#     if i > 20:
#         break
#     answer += all_nodes[i]
#     if i != len((all_nodes)) and i != 20:
#         answer += "、"



import csv

import pandas as pd
import streamlit as st
from streamlit_chatbox import *
from streamlit_agraph import agraph, Edge, Config,Node
from py2neo import Graph, Relationship, NodeMatcher, Subgraph,RelationshipMatcher



def readcsv(index_association_file):
    df = pd.read_csv(index_association_file)

    with open(index_association_file) as csv_file:
        # creating an object of csv reader
        # with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter=',')
        # list to store the names of columns
        list_of_column_names = []
        list_of_row_names = []
        # loop to iterate through the rows of csv
        for col in csv_reader:
            # adding the first row
            list_of_column_names.append(col)
            # breaking the loop after the
            # first iteration itself
            break

    with open(index_association_file,'r') as csv_file:
        # creating an object of c
        list_of_row_names = []
        reader = csv.reader(csv_file)
        result = list(reader)

        for i in range(len(result)):

            list_of_row_names.append(result[i][0])

    list_of_column_names[0] = list_of_column_names[0][1:]
    list_of_row_names = list_of_row_names[1:]
    # printing the result
    # print("List of column names : ",
    #       list_of_column_names[0])
    # print("List of row names : ",
    #       list_of_row_names)
    return list_of_column_names[0],list_of_row_names






def kg_graph(nodes, edges):
    config = Config(width=500,
                    height=500,
                    directed=False, 
                    physics=True, 
                    hierarchical=False,
                    # **kwargs
                    )
    return agraph(nodes=nodes, edges=edges, config=config)

def getSimData(name,option,num,rate):
    m_sim = pd.read_csv("../data2/microbe_features.txt", header=None, index_col=None, sep='\t').values
    d_sim = pd.read_csv("../data2/disease_features.txt", header=None, index_col=None, sep='\t').values
    micro, dis = readcsv("../data2/index_association.csv")
    data = []

    if option=='疾病':
        index = dis.index(name)
        k = 0
        for i in range(len(d_sim)):
            if k >= num:
                break
            if d_sim[index][i]>rate:
                if dis[i] != name:
                    data.append(dis[i])
                    k += 1

    elif option =="微生物":
        index = micro.index(name)
        k = 0
        for i in range(len(m_sim)):
            if m_sim[index][i] > rate:
                if k >= num:
                    break
                if micro[i] != name:
                    data.append(micro[i])
                    k += 1
    data_df = pd.DataFrame(data)
    return data_df

chat_box = ChatBox(
    assistant_avatar="../qianduan/imgs/robot.png",
    user_avatar="../qianduan/imgs/user.png",
    greetings=[':rose: **Hi!** **你好呀！** :rose:',
                ':robot_face: :rainbow[我是一个基于疾病与微生物关联预测的知识图谱的问答机器人，我有很多功能:]',
                ' :one: **疾病与微生物关联知识图谱问答功能** :woman-gesturing-ok: ',
                ' :two: **相关部分知识图谱展示** :man-gesturing-ok:',
                ':orange[**——————————————————————————————————**]',
                ':robot_face: :rainbow[我的知识中包含了842种疾病节点种微生物节点以及33104种节点关系]',
                ':robot_face: :rainbow[请注意，在向我提问时请使用英文的疾病名称] :rose:',
                ':orange[**——————————————————————————————————**]']
        )


#st.sidebar - 在侧边栏增添交互元素
with st.sidebar:
    st.header('微生物与疾病知识图谱问答系统')
    tab1, tab2 = st.tabs(["模糊搜索","知识图谱"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            option = st.selectbox('搜索内容', ['疾病', '微生物'],index=1)
        with col2:
            name = st.text_input('请在此输入名称: ')
        col3, col4 = st.columns(2)
        with col3:
            num = st.number_input(label='查询数量', min_value=0, max_value=10, value=3, step=1)
        with col4:
            rate = st.number_input(label='相似度', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        if st.button('搜索'):
            df = getSimData(name,option,num,rate)
            # gb = GridOptionsBuilder.from_dataframe(df)
            # gb.configure_default_column(editable=True, groupable=True)  # 默认列配置
            # go = gb.build()
            # AgGrid(df, gridOptions=go, height=210, fit_columns_on_grid_load=True,reload_data=False)
            st.dataframe(df)


chat_box.init_session()
chat_box.output_messages()

if query := st.chat_input('请输入您想问的疾病的名字……'):
    chat_box.user_say(query) #显示用户输入
    graph = Graph('http://localhost:7474', auth=("neo4j", "neo4j"))
    node_matcher = NodeMatcher(graph)
    disease_node_exist = node_matcher.match("Disease", name=query).exists()
    if disease_node_exist == False:
        print("不存在该疾病")

    relationship_matcher = RelationshipMatcher(graph)
    centerNode = node_matcher.match("Disease").where(name=query).first()
    new_relationship = list(relationship_matcher.match((centerNode, None), "new_find"))
    already_relationship = list(relationship_matcher.match((centerNode, None), "already_exists"))
    print(len(already_relationship))



    for j in range(len(new_relationship)):
        sign = False
        for i in range(len(new_relationship) - 1 - j):
            if new_relationship[i]["sim_score"]< new_relationship[i + 1]["sim_score"]:
                t = new_relationship[i]
                new_relationship[i] = new_relationship[i + 1]
                new_relationship[i + 1] = t
                sign = True
        if not sign:
            break

    for i in range(len(new_relationship)):
        print(new_relationship[i]["sim_score"])
    print(len(new_relationship))


    all_nodes = []
    nodes = []
    edges = []
    nodes.append(
        Node(id=query,
             label=query,
             size=25,
             color='#9370DB')
    )
    csv_relationship = []
    num_micro=0
    for i in range(len(new_relationship)):
        if num_micro>20:
            break
        label = str('{:.2f}'.format(new_relationship[i]["sim_score"]))
        all_nodes.append(new_relationship[i].end_node["name"])
        nodes.append(Node(id=new_relationship[i].end_node["name"],
                          label=new_relationship[i].end_node["name"],
                          size=15,
                          color='#BA55D3'))
        edges.append(Edge(source=query,
                          label=label,
                          target=new_relationship[i].end_node["name"],
                          color="red",
                          # **kwargs
                          )
                         )
        num_micro +=1
        csv_relationship.append(new_relationship[i])

    for i in range(len(already_relationship)):
        label = str('{:.2f}'.format(already_relationship[i]["sim_score"]))
        all_nodes.append(already_relationship[i].end_node["name"])
        nodes.append(Node(id=already_relationship[i].end_node["name"],
                              label=already_relationship[i].end_node["name"],
                              size=15,
                              color='#BA55D3'))
        edges.append(Edge(source=query,
                              label=label,
                              target=already_relationship[i].end_node["name"],
                              color="blue",
                              # **kwargs
                              )
                         )
        csv_relationship.append(already_relationship[i])

    answer = "与该疾病相关的微生物有："
    answer_drug = "对于该疾病可使用一下治疗药物："
    for i in range(len(all_nodes)):
        answer += all_nodes[i]
        if i != len((all_nodes)):
            answer += "、"
        else:
            answer += "。"


    elements = chat_box.ai_say(  #显示机器人输入
        [
            Markdown(answer, in_expander=False,
                     expanded=True, title="answer"),
        ]
    )

    # for j in range(len(csv_relationship)):
    #     sign = False
    #     for i in range(len(csv_relationship) - 1 - j):
    #         if csv_relationship[i]["sim_score"]< csv_relationship[i + 1]["sim_score"]:
    #             t = csv_relationship[i]
    #             csv_relationship[i] = csv_relationship[i + 1]
    #             csv_relationship[i + 1] = t
    #             sign = True
    #     if not sign:
    #         break

    # with open("D:\\mycode\\micro-dis\\qianduan\\relationship.csv","w") as f:
    #     file_writer = csv.writer(f)
    #     row = ["name","sim","type"]
    #     file_writer.writerow(row)
    #
    # for i in range(len(csv_relationship)):
    #     name = csv_relationship[i].end_node["name"]
    #     sim = csv_relationship[i]["sim_score"]
    #     typea = type(csv_relationship[i]).__name__
    #     with open("D:\\mycode\\micro-dis\\qianduan\\relationship.csv", "a+") as f:
    #         file_writer = csv.writer(f)
    #         row = [name, sim, typea]
    #         file_writer.writerow(row)




    #给出知识图谱的内容
    with st.sidebar:
        with tab2:
            config = Config(width=750,
                            height=950,
                            directed=True,
                            physics=True,
                            hierarchical=False,
                            # **kwargs
                            )
            agraph(nodes=nodes,
                   edges=edges,
                   config=config)
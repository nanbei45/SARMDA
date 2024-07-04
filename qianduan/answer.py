from py2neo import Graph
from streamlit_agraph import agraph, Node, Edge, Config
import difflib

#知识图谱连接参数
uri = "http://localhost:7474"
username = "neo4j"
password = "neo4j"

class answer_from_robot():
    def __init__(self, en_dict, right_name, micro_li):
        self.graph = Graph(uri, auth=(username,password))
        self.en_dict = en_dict
        self.right_name = right_name
        self.micro_li = micro_li
        self.answer_list, self.all_nodes, self.all_edges = self.answer_ques(self.en_dict, self.right_name)
    
    def answer_ques(self, IR, en_dict, right_name):
        easy_ques = ['recause']
        node = []
        edge = []
        all_nodes = []
        all_edges = []
        all_answers = []
        if not IR:
            all_answers.append('脑子炸了呀，没明白您的意思o(╥﹏╥)o')
        for ir in IR:
            if ir in easy_ques:
                answers, nodes, edges, node, edge = self.esay_answers(en_dict, right_name, ir, node, edge)
                all_answers.extend(answers)
                all_nodes.extend(nodes)
                all_edges.extend(edges)

        return all_answers, all_nodes, all_edges

    def esay_answers(self, en_dict, right_name, relation, node, edge):#导演、演员、编剧、上映地区、类型、上映时间、语言
        nodes = []
        edges = []
        color = 'yellow'
        if relation == 'recause':
            color = 'blue'
        answers = []
        if en_dict['dis']:
            for n in en_dict['dis']:
                if n in right_name:
                    answer = self.graph.run("MATCH (:Disease {name:'" + n + "'})-[:"+relation+"]-(p) RETURN p").data()
                    if n not in node:
                        nodes.append(Node(id=n, 
                                    label=n, 
                                    size=25, 
                                    color='green',))
                        node.append(n)
                    if not answer:
                        www = self.graph.run("MATCH (m:Disease{name:'" + n + "'}) RETURN m.url").data()
                        answers.append('暂时缺少n'+relation+'相关信息，详情可以通过下面的网址进行查询：')
                        answers.append(n+' : '+www[0]['m.url'])
                    else:
                        r = ''
                        for an in answer:
                            r += an['p']['name'] + ' '
                            if an['p']['name'] not in node:
                                nodes.append(Node(id=an['p']['name'], 
                                            label=an['p']['name'], 
                                            size=25, 
                                            color=color,))
                                node.append(an['p']['name'])
                            if (n, an['p']['name']) not in edge:
                                edges.append(Edge(source=n, 
                                            label=relation, 
                                            target=an['p']['name'], ))
                                edge.append((n, an['p']['name']))
                        answers.append('n'+'的'+relation+'：' + r)

                else:
                    answers.append('暂时缺少n'+relation+'相关信息，请核对疾病名')

        return answers, nodes, edges, node, edge

    def reco_answers(self):
        return ['功能未开通']


    

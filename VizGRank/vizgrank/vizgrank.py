import sys
from VizGRank.dp_pack import deepeye
from VizGRank.dp_pack.table import Table
from VizGRank.dp_pack.instance import Instance
from VizGRank.dp_pack.instance import ViewPosition
from VizGRank.dp_pack.view import View
import networkx as nx
import numpy as np
from VizGRank.vizgrank.node_relation import context_dissimilarity, context_similarity


class VizGRank:
    def __init__(self, name, G=None, base_alg='PageRank', relation_func=context_similarity, personalization_func=None, authority=0,
                 decay=None,
                 parallel=False):
        self.name = name
        self.parallel = parallel
        self.raw_table = Table(None, None, None, None)
        self.instance = Instance(self.raw_table)
        self.base_alg = base_alg
        self.node_relation = relation_func
        self.personalization_func = personalization_func
        self.personalization = None
        self.autority = authority
        self.model_name = ''
        self.decay = decay
        self.G = G
        self.ordered = []
        self.dataset_name = ''
        self.dp: deepeye = ''
    
    def read_data(self, filepath, dtypes=None, mode='csv'):
        if mode == 'csv':
            # for deepeye
            dp_1 = deepeye('demo')
            self.dp = dp_1
            dp_1.from_csv(filepath, types=dtypes)
            instance = Instance(self.dataset_name)
            instance.addTable(Table(instance, False, '', ''))  # 'False'->transformed '',''->no describe yet
            self.instance = dp_1.csv_handle(instance)
            dp_1.instance = self.instance
        elif mode == 'mysql':
            pass
        elif mode == 'kylin':
            pass
        return self
    
    def generate_visualizations(self):
        if len(self.instance.tables[0].D) == 0:
            print('no data in table')
            sys.exit(0)
        # print(instance.table_num, instance.view_num)
        self.instance.addTables(self.instance.tables[0].dealWithTable())  # the first deal with is to transform the table into several small ones
        # print(instance.table_num, instance.view_num)
        del self.instance.tables[0].D
        del self.instance.tables[0].features
        begin_id = 1
        while begin_id < self.instance.table_num:
            self.instance.tables[begin_id].dealWithTable()  # to generate views
            del self.instance.tables[begin_id].D
            del self.instance.tables[begin_id].features
            begin_id += 1
        if self.instance.view_num == 0:
            print('no chart generated')
            sys.exit(0)
        # print(instance.table_num, instance.view_num)s
        return self
    
    def rank_visualizations(self):
        self.getScore()
        return self
    
    def getScore(self):  # 这个是我要改进的函数 尤其重要
        instance = self.instance
        instance.getM()
        instance.getW()
        score = instance.getScoreArr()
        instance.views = []
        for i in range(instance.table_num):
            instance.views.extend([ViewPosition(i, view_pos) for view_pos in range(instance.tables[i].view_num)])
        personalization = {}
        if not self.G:
            G = [[0 for i in range(instance.view_num)] for j in range(instance.view_num)]
            for i in range(instance.view_num):
                view_i: View = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                view_i.view_id = i
                for j in range(i + 1, instance.view_num):
                    view_j = instance.tables[instance.views[j].table_pos].views[instance.views[j].view_pos]
                    view_j.view_id = j
                    G[j][i] = G[i][j] = self.node_relation(view_i, view_j)
                personalization[i] = self.personalization_func(
                    view_i) if self.personalization_func and self.personalization_func != Instance.getScoreArr else 0
                view_i.per = personalization[i]
            self.personalization = personalization if self.personalization_func else None
            if self.personalization_func == Instance.getScoreArr:
                self.personalization = dict(enumerate(score))
            self.G = G
        
        personalization = self.personalization if self.personalization_func else None
        prG = nx.from_numpy_matrix(np.array(self.G))
        
        pr = nx.pagerank_numpy(prG, personalization=personalization)
        score = pr
        
        for i in range(instance.view_num):
            instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos].score = score[i]
        instance.views = self.ordered = sorted(instance.views, key=lambda view: instance.tables[view.table_pos].views[view.view_pos].score,
                                               reverse=True)
    
    def to_list_benchmark_a(self):
        instance = self.instance
        export_list = []
        order1 = order2 = 1
        old_view = ''
        for i in range(instance.view_num):
            view = instance.views[i]
            if old_view:
                order2 = 1
                order1 += 1
            export_list.append(view.output_benchmark(order1))
            old_view = view
        output = {
            'charts': export_list
        }
        return output
    
    def to_list_benchmark(self):
        instance = self.instance
        export_list = []
        order1 = order2 = 1
        old_view = ''
        for i in range(instance.view_num):
            view = instance.tables[self.ordered[i].table_pos].views[self.ordered[i].view_pos]
            if old_view:
                order2 = 1
                order1 += 1
            export_list.append(view.output_benchmark(order1))
            old_view = view
        output = {
            'charts': export_list
        }
        return output
    
    def to_list_benchmark_raw(self):
        instance = self.instance
        export_list = []
        order1 = order2 = 1
        old_view = ''
        for i in range(instance.view_num):
            view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
            if old_view:
                order2 = 1
                order1 += 1
            export_list.append(view.output_benchmark(order1))
            old_view = view
        output = {
            'charts': export_list
        }
        return output
    
    def output_visualizations(self):
        instance = self.instance
        export_list = []
        order1 = order2 = 1
        old_view = ''
        for i in range(instance.view_num):
            view = instance.tables[self.ordered[i].table_pos].views[self.ordered[i].view_pos]
            if old_view:
                order2 = 1
                order1 += 1
            export_list.append(view.output_visrank(order1))
            old_view = view
        return export_list
    
    def load_visualizations_from_file(self, dataset_name, types, input_dir, input_json):
        pass
    
    def to_html(self):
        self.dp.rank_method = 'partial_order'  # = 'partial_order'
        self.dp.to_single_html()
        print('Output file data_all.html')

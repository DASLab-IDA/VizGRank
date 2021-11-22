"""
A copy from https://github.com/Thanksyy/DeepEye-APIs with deletion and modification.
Origin: Yuyu Luo
"""

import math
import random

import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from numpy import corrcoef
from .features import Type
from functools import reduce
from scipy import stats
from . import global_variables


class Chart(object):
    bar = 0
    line = 1
    scatter = 2
    pie = 3
    chart = ['bar', 'line', 'scatter', 'pie']


class View(object):
    """
    Attributes:
        table(Table): the table corresponding to this view.
        fx(Feature): the attributes of axis x.
        fy(Feature): the attributes of axis y.
        x_name(str): the name of axis x.
        y_name(str): the name of axis y.
        series_num(int): the number of classification.
        X(list): the data of axis x.
        Y(list): the data of axis y.
        chart(str): the type of the chart, including bar, line ,scatter ans pie.
        tuple_num(int): tuple_num in the corresponding table (the number of columns after transformation).
        score_l(float): the score of the chart in learning_to_rank method.
        M(float): M value in the paper.
        Q(float): Q value in the paper.
        score(float): the score of the chart in partial_order method.
    """
    
    def __init__(self, table, x_id, y_id, z_id, series_num, X, Y, chart):
        self.per = 0
        self.table = table
        self.fx = table.features[x_id]
        self.fy = table.features[y_id]
        self.x_name_origin = global_variables.columns[self.fx.origin]
        self.y_name_origin = global_variables.columns[self.fy.origin]
        self.x_name = self.fx.name
        self.y_name = self.fy.name
        self.trans_x = "GROUP"
        self.trans_y = "CNT"
        self.trans_z = "GROUP"
        self.z_id = z_id
        self.z_name_origin = global_variables.columns[self.z_id] if z_id != -1 else ''
        self.series_num = series_num
        self.X = X
        self.Y = Y
        self.chart = chart
        self.tuple_num = table.tuple_num
        self.score_l = 0  # learning_to_rank score
        self.M = self.Q = self.W = self.score = 0  # partial and div_ranking score
        self.getM()
        self.getQ()
        self.getTrans()
    
    def getTrans(self):
        self.trans_y = self.y_name.split('(')[0] if self.y_name.startswith('CNT(') or self.y_name.startswith('SUM(') or self.y_name.startswith(
            'AVG(') \
            else "raw"
        if self.z_id != -1:
            self.trans_z = self.table.describe1.split(' ')[0]
            self.trans_x = self.table.describe2.split(' ')[0] if self.table.describe2 else "raw"
        else:
            self.trans_x = self.table.describe1.split(' ')[0] if self.table.describe1 else "raw"
    
    #### function for learning to rank
    def getCorrelation_l(self, series_id):
        """
        Calculate correlation coefficient of X and Y, log(X) and Y, X and log(Y), log(X) and log(Y)
        to determine the relationship of X and Y such as linear, exponential, logarithm and power.
        (especially for learning_to_rank method)

        Args:
            series_id(int): the index of X and Y(list), determining correlation coefficient of which
                            two columns are to be calculated.
            
        Returns:
            result(float): For the correlation coefficient of X and Y, log(X) and Y, X and log(Y),
                           log(X) and log(Y), result is the max of the four correlation coefficient.
            
        """
        if self.fx.type == Type.categorical:  # regard the corrcoef of categorical as 0
            return 0
        if self.fx.type == Type.temporal:
            data1 = [i for i in range(self.tuple_num // self.series_num)]
        else:
            data1 = self.X[series_id]
        data2 = self.Y[series_id]
        log_data1 = log_data2 = []
        if self.fx.type != Type.temporal and self.fx.min != '' and self.fx.min > 0:
            log_data1 = map(math.log, data1)
        if self.fy.min != '' and self.fy.min > 0:
            log_data2 = map(math.log, data2)
        result = 0
        # calculate and compare correlation
        # linear
        try:
            result = abs(corrcoef(data1, data2)[0][1])
        except Exception as e:
            result = 0
        else:
            pass
        
        # exponential
        if log_data2:
            try:
                r = abs(corrcoef(data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass
        
        # logarithm
        if log_data1:
            try:
                r = abs(corrcoef(log_data1, data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass
        
        # power
        if log_data1 and log_data2:
            try:
                r = abs(corrcoef(log_data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                result = 0
            else:
                pass
        if not -1 <= result <= 1:
            result = 0
        return result
    
    def output_score(self):
        """
        For learning_to_rank method, get score of each chart and write to files

        Args:
            None
            
        Returns:
            The string which needs to be written in .score file
            
        """
        correlation = max([self.getCorrelation_l(i) for i in range(self.series_num)])
        if self.fx.min == '':
            self.fx.min = 0
        if self.fy.min == '':
            self.fy.min = 0
        if self.fx.type == Type.temporal:
            return '1 qid:1 1:' + str(self.fx.type) + ' 2:' + str(self.fy.type) + ' 3:' + str(
                self.tuple_num) + ' 4:' + str(self.tuple_num) + ' 5:0 6:' + str(self.fy.min) + ' 7:0 8:' + str(
                self.fy.max) + ' 9:' + str(self.fx.distinct) + ' 10:' + str(self.fy.distinct) + ' 11:' + str(
                self.fx.ratio) + ' 12:' + str(self.fy.ratio) + ' 13:' + str(correlation) + ' 14:' + str(self.chart)
        else:
            return '1 qid:1 1:' + str(self.fx.type) + ' 2:' + str(self.fy.type) + ' 3:' + str(
                self.tuple_num) + ' 4:' + str(self.tuple_num) + ' 5:' + str(self.fx.min) + ' 6:' + str(
                self.fy.min) + ' 7:' + str(self.fx.max) + ' 8:' + str(self.fy.max) + ' 9:' + str(
                self.fx.distinct) + ' 10:' + str(self.fy.distinct) + ' 11:' + str(self.fx.ratio) + ' 12:' + str(
                self.fy.ratio) + ' 13:' + str(correlation) + ' 14:' + str(self.chart)
    
    #### function for partial order and diversified ranking
    def getCorrelation(self, series_id):
        """
        Calculate correlation coefficient of X and Y, log(X) and Y, X and log(Y), log(X) and log(Y)
        to determine the relationship of X and Y such as linear, exponential, logarithm and power.
        (especially for partial order and diversified ranking methods)

        Args:
            series_id(int): the index of X and Y(list), determining correlation coefficient of which
                            two columns are to be calculated.
            
        Returns:
            result(float): For the correlation coefficient of X and Y, log(X) and Y, X and log(Y),
                           log(X) and log(Y), result is the max of the four correlation coefficient.
            
        """
        if self.fx.type == Type.temporal:
            data1 = [i for i in range(self.tuple_num // self.series_num)]
        else:
            if series_id < len(self.X):
                data1 = self.X[series_id]
        data2 = self.Y[series_id]
        log_data1 = log_data2 = []
        if self.fx.type != Type.temporal and self.fx.min != '' and self.fx.min > 0:
            log_data1 = map(math.log, data1)
        if self.fy.minmin != '' and self.fy.minmin > 0:
            log_data2 = map(math.log, data2)
        log_data2 = map(math.log, data2)
        result = 0
        # linear
        try:
            result = abs(corrcoef(data1, data2)[0][1])
        except Exception as e:
            result = 0
        # else:
        #     pass
        
        # exponential
        if log_data2:
            try:
                r = abs(corrcoef(data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass
                # print("2 ", e)
                # result = 0
            # else:
            #     pass
        
        # logarithm
        if log_data1:
            try:
                r = abs(corrcoef(log_data1, data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass
                # print("3 ", e)
                # result = 0
            else:
                pass
        
        # power
        if log_data1 and log_data2:
            try:
                r = abs(corrcoef(log_data1, log_data2)[0][1])
                if r > result:
                    result = r
            except Exception as e:
                pass
                # print("4 ", e)
                # result = 0
            else:
                pass
        
        return result
    
    def getM(self):
        """
        Calculate M value in the paper

        Args:
            None
            
        Returns:
            None
            
        """
        # self.tuple_num: x数目 * series系列数目
        # self.series_num: 系列数目
        # self.table.instance.tuple_num: 数据总行数
        # self.Y[0]: 第一组数据的值
        if self.chart == Chart.pie:
            if self.tuple_num == 1:
                self.M = 0
            elif 2 <= self.tuple_num <= 10:
                sumY = sum(self.Y[0])
                self.M = reduce(lambda x, y: x + y, map(lambda y: -(1.0 * y / sumY) * math.log(1.0 * y / sumY), self.Y[0]))
            elif self.tuple_num > 10:
                sumY = sum(self.Y[0])
                self.M = reduce(lambda x, y: x + y, map(lambda y: -(1.0 * y / sumY) * math.log(1.0 * y / sumY), self.Y[0])) * 10.0 / (self.tuple_num)
        elif self.chart == Chart.bar:
            if self.tuple_num // self.series_num == 1:
                self.M = 0
            elif 2 <= self.tuple_num // self.series_num <= 20:
                self.M = 1
                # self.M = (max(self.Y[0]) - min(self.Y[0])) / (sum(self.Y[0]) / float(self.tuple_num / self.series_num))
            else:
                self.M = 20 / (self.tuple_num // self.series_num)
                # self.M = 10.0 / self.tuple_num * ((max(self.Y[0]) - min(self.Y[0])) / (sum(self.Y[0]) / float(self.tuple_num / self.series_num)))
        elif self.chart == Chart.scatter:
            if self.series_num == 1:
                self.M = self.getCorrelation(0)
            else:
                self.M = max([self.getCorrelation(i) for i in range(self.series_num)])
        else:  # if self.chart == Chart.line
            if self.series_num == 1:
                if self.getCorrelation(0) > 0.3:
                    self.M = 1
                else:
                    self.M = 0
            else:
                if max([self.getCorrelation(i) for i in range(self.series_num)]) > 0.3:
                    self.M = 1
                else:
                    self.M = 0
    
    def getQ(self):
        """
        Calculate Q value in the paper

        Args:
            None
            
        Returns:
            None
            
        """
        # self.Q = 1
        # if self.chart == Chart.bar or self.chart == Chart.pie:
        self.Q = 1 - 1.0 * (self.tuple_num / self.series_num) / self.table.instance.tuple_num
    
    def output_visrank(self, order):
        """
            Encapsulate the value of several variables in variable data(ruturned value).

        Args:
            order(int): Not an important argument, only used in the assignment of data.

        Returns:
            data(str): A string including the value of several variables:
                       order1, order2, describe, x_name, y_name, chart, classify, x_data, y_data.

        """
        classify = []
        if self.series_num > 1:
            classify = list(map(lambda x: x[0], self.table.classes))
        x_data = self.X
        if self.fx.type == Type.numerical:
            x_data = self.X
        elif self.fx.type == Type.categorical:
            x_data = self.X
        else:
            for i in range(len(self.X)):
                self.X[i] = list(map(lambda x: str(x), self.X[i]))
        y_data = self.Y
        if self.fy.type == Type.numerical:
            y_data = self.Y
        elif self.fy.type == Type.categorical:
            y_data = self.Y
        else:
            for i in range(len(self.Y)):
                self.Y[i] = list(map(lambda y: str(y), self.Y[i]))
        
        data = {
            'order1': order,
            'order2': 1,
            'score': self.score,
            'preference': self.per,
            'describe': self.table.describe,
            'x_name': self.x_name,
            'y_name': self.y_name,
            'chart': Chart.chart[self.chart],
            'classify': classify,
            'x_data': x_data,
            'y_data': y_data
        }
        return data
    
    #### function for all
    def output(self, order):
        """
            Encapsulate the value of several variables in variable data(ruturned value).

        Args:
            order(int): Not an important argument, only used in the assignment of data.
            
        Returns:
            data(str): A string including the value of several variables:
                       order1, order2, describe, x_name, y_name, chart, classify, x_data, y_data.
            
        """
        classify = str([])
        if self.series_num > 1:
            classify = str([v[0] for v in self.table.classes]).replace("u'", '\'').replace("'", '"')
        x_data = str(self.X)
        if self.fx.type == Type.numerical:
            x_data = str(self.X).replace("'", '').replace('"', '').replace('L', '')
        elif self.fx.type == Type.categorical:
            x_data = str(self.X).replace("u'", '\'').replace("'", '"')
        else:
            len_x = len(self.X)
            # x_data = '[' + reduce(lambda s1, s2: s1 + s2, [str(map(str, self.X[i])) for i in range(len_x)]).replace("'",'"') + ']'
            x_data = '["%s"]' % ''.join(
                list(reduce(lambda s1, s2: s1 + s2, ['","'.join(list(map(str, self.X[i]))) for i in range(len_x)]).replace("'", '"')))
        y_data = str(self.Y)
        if self.fy.type == Type.numerical:
            y_data = str(self.Y).replace("'", '').replace('"', '').replace('L', '')
        elif self.fy.type == Type.categorical:
            y_data = str(self.Y).replace("u'", '\'').replace("'", '"')
        else:
            len_y = len(self.Y)
            # x_data = '[' + reduce(lambda s1, s2: s1 + s2, [str(map(str, self.X[i])) for i in range(len_x)]).replace("'",'"') + ']'
            y_data = '["%s"]' % ''.join(
                list(reduce(lambda s1, s2: s1 + s2, ['","'.join(list(map(str, self.Y[i]))) for i in range(len_y)]).replace("'", '"')))
        # if self.fy.type == Type.numerical:
        #    y_data = y_data.replace('L', '')
        data = '{"order1":' + str(order) + ',"order2":' + str(
            1) + ',"describe":"' + self.table.describe + '","x_name":"' + self.fx.name + '","y_name":"' + self.fy.name + '","chart":"' + Chart.chart[
                   self.chart] + '","classify":' + classify + ',"x_data":' + x_data + ',"y_data":' + y_data + '}'
        # data = 'score:' + str(round(self.score, 2)) + '\tM:' + str(round(self.M, 2)) + '\tQ:' + str(round(self.Q, 2)) + '\tW:' + str(round(
        # self.W, 2)) + '{"order":' + str(order) + ',"describe":"' + self.table.describe + '","x_name":"' + self.fx.name + '","y_name":"' +
        # self.fy.name + '","chart":"' + Chart.chart[self.chart] + '","classify":' + classify + ',"x_data":' + x_data + ',"y_data":' + y_data + '}'
        return data
    
    def output_benchmark(self, order=None):
        if not order:
            data = {
                'type': Chart.chart[self.chart]
            }
        else:
            data = {
                'rank': order,
                'type': Chart.chart[self.chart],
                'score': self.score
            }
        
        transs = self.table.describe1.split(' ') if not self.table.describe2 else self.table.describe2.split(' ')
        trans_x = 'raw'
        if transs[0] == 'BIN':
            trans_x = transs[-1].lower()
            if transs[-1] == 'WEEKDAY':
                trans_x = 'week'
        
        trans_y = 'raw'
        if self.y_name[0:3] == 'SUM':
            trans_y = 'sum'
        elif self.y_name[0:3] == 'CNT':
            trans_y = 'count'
        elif self.y_name[0:3] == 'AVG':
            trans_y = 'mean'
        
        if Chart.chart[self.chart] == 'bar':
            x = {
                'attr': self.x_name_origin,
                'trans': trans_x,
                'groupby': True
            }
            y = {
                'attr': self.y_name_origin,
                'trans': trans_y,
                'calcus': 'raw',
            }
            data['channels'] = {
                'x': x,
                'y': y,
            }
            if self.z_id != -1:
                color = {
                    'attr': self.z_name_origin,
                    'trans': 'raw',
                    'groupby': True
                }
                data['channels']['color'] = color
        elif Chart.chart[self.chart] == 'line':
            x = {
                'attr': self.x_name_origin,
                'trans': trans_x,
                'groupby': True
            }
            y = {
                'attr': self.y_name_origin,
                'trans': trans_y,
                'calcus': 'raw',
            }
            data['channels'] = {
                'x': x,
                'y': y,
            }
            if self.z_id != -1:
                color = {
                    'attr': self.z_name_origin,
                    'trans': 'raw',
                    'groupby': True
                }
                data['channels']['color'] = color
        elif Chart.chart[self.chart] == 'pie':
            x = {
                'attr': self.x_name_origin,
                'trans': trans_x,
                'groupby': True
            }
            y = {
                'attr': self.y_name_origin,
                'trans': trans_y,
                'calcus': 'percentage',
            }
            data['channels'] = {
                'x': x,
                'y': y
            }
        elif Chart.chart[self.chart] == 'scatter':
            x = {
                'attr': self.x_name_origin,
                'trans': 'raw'
            }
            y = {
                'attr': self.y_name_origin,
                'trans': 'raw'
            }
            data['channels'] = {
                'x': x,
                'y': y
            }
            if self.z_id != -1:
                color = {
                    'attr': self.z_name_origin,
                    'trans': 'raw',
                    'groupby': True
                }
                data['channels']['color'] = color
        
        return data
    
    def getSimilarityWith(self, view_j):
        viz_a = []
        viz_b = []
        
        # get all columns
        columns_a = [(self.y_name_origin, self.trans_y), (self.x_name_origin, self.trans_x)]
        if self.z_id != -1:
            columns_a.append((self.z_name_origin, self.trans_z))
        
        columns_b = [(view_j.y_name_origin, view_j.trans_y), (view_j.x_name_origin, view_j.trans_x)]
        if self.z_id != -1:
            columns_b.append((view_j.z_name_origin, view_j.trans_z))
        
        viz_a.extend(columns_a)
        viz_b.extend(columns_b)
        
        # get chart type
        viz_a.append(Chart.chart[self.chart])
        viz_b.append(Chart.chart[view_j.chart])
        
        set1, set2 = set(viz_a), set(viz_b)
        similarity = len(set1 & set2) / float(len(set1 | set2))
        
        return similarity
    
    def getColumnCorrelation(self):
        corr = 0
        if self.chart == Chart.pie:
            corr = stats.entropy(self.Y[0])
        elif self.chart == Chart.bar:
            if self.series_num == 1:
                corr = stats.entropy(self.Y[0])
            else:
                f, p = stats.f_oneway(*self.Y)
                corr = 1 - p
        elif self.chart == Chart.scatter or self.chart == Chart.line:
            if self.series_num == 1:
                corr = self.getCorrelation(0)
            else:
                corr = max([self.getCorrelation(i) for i in range(self.series_num)])
        
        if math.isnan(corr) or math.isinf(corr):
            corr = 0
        
        return corr
    
    def get_transformed_fields(self):
        x = [y for x in self.X for y in x]
        y = [y for x in self.Y for y in x]
        raw_x = [(i, x[i]) for i in range(len(x))]
        raw_y = [(i, y[i]) for i in range(len(y))]
        fields = {self.x_name_origin + ' ' + self.x_name + ' ' + self.trans_x + ' x': dict(raw_x),
                  self.y_name_origin + ' ' + self.y_name + ' ' + self.trans_y + ' y': dict(raw_y)}
        
        if self.z_id != -1:
            raw_z = list(enumerate(list(map(lambda x: x[0], self.table.classes))))
            fields[self.z_name_origin + ' ' + self.trans_z + ' z'] = dict(raw_z)
        
        return fields
    
    def get_transformed_types(self):
        ts = ['none', 'categorical', 'numerical', 'temporal']
        ts_dic = {
            'categorical': 'object',
            'temporal': 'object',
            'numerical': 'float32'
        }
        types = {self.x_name_origin + ' ' + self.x_name + ' ' + self.trans_x + ' x': ts_dic[ts[self.fx.type]],
                 self.y_name_origin + ' ' + self.y_name + ' ' + self.trans_y + ' y': ts_dic[ts[self.fy.type]]}
        
        if self.z_id != -1:
            types[self.z_name_origin + ' ' + self.trans_z + ' z'] = ts_dic[ts[self.table.instance.tables[0].types[self.z_id]]]
        
        return types
    
    def get_raw_fields(self):
        raw_table = self.table.instance.tables[0]
        raw_x = [(i, raw_table.D[i][self.fx.origin]) for i in range(len(raw_table.D))]
        raw_y = [(i, raw_table.D[i][self.fy.origin]) for i in range(len(raw_table.D))]
        fields = {self.x_name_origin: dict(raw_x),
                  self.y_name_origin: dict(raw_y)}
        
        if self.z_id != -1:
            raw_z = [(i, raw_table.D[i][self.z_id]) for i in range(len(raw_table.D))]
            fields[self.z_name_origin] = dict(raw_z)
        
        return fields
    
    def get_raw_types(self):
        raw_table = self.table.instance.tables[0]
        types = {self.x_name_origin: raw_table.dtypes[self.fx.origin].name,
                 self.y_name_origin: raw_table.dtypes[self.fy.origin].name}
        
        if self.z_id != -1:
            types[self.z_name_origin] = raw_table.dtypes[self.z_id].name
        
        return types

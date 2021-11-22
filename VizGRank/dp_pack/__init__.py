"""
A copy from https://github.com/Thanksyy/DeepEye-APIs with deletion and modification.
Origin: Yuyu Luo
"""

# coding:utf-8
import os
import sys

import numpy as np
import pandas as pd
import datetime
from pprint import *
from pyecharts.charts import Bar, Line, Scatter, Pie, Grid, Page
from pyecharts import options as opts

from KaggleBench.benchmark_manager.util.monitor import monitor
from .instance import Instance
from .table import Table
from .view import Chart
from .features import Type

import re  # use regular expressions when recognizing the date type

from IPython.core.display import display, HTML

methods_of_import = ['none', 'mysql', 'csv']
methods_of_ranking = ['none', 'learn_to_rank', 'partial_order', 'diversified_ranking']
from . import global_variables


class default(object):
    def __init__(self):
        return


class deepeye(object):
    """
    Attributes:
        name(str): the name of the deepeye project, default set as "deepeye".
        is_table_info(bool): Whether or not the table info has been completely set.
        import_method(str): the method of import, including "mysql" and "csv".
        rank_method(str): the method of rank, including "learn_to_rank", "partial_order" and "diversified_ranking"
        table_name(str): the name of the table to be visualized.
        column_names(list): the name of each column in the table.
        column_types(list): the type of each column in the table.
        csv_path(str): the path of the csv dataset to be visualized.
        csv_dataframe(pandas.Dataframe): dataset of Dataframe format.
        instance(Instance): an Instance object corresponding to the dataset.
        page(Page): the page to be showed in html or jupyter notebook.
    """
    
    ##### initial function
    def __init__(self, *name):
        """
        Initialize the table infomation: name, istable, importmethod, rankmethod
        There are two ways of import method: mysql and csv.(only csv can use yet)
        Three ways of ranking: learn_to_rank(rank by machine learning)
                               partial_order(rank by rules of expert knowledge)
                               diversified_ranking(hybrid of the two methods)

        Args:
            name(str): The name of the table(s)

        Returns:
            None

        """
        if not name:
            self.name = 'deepeye'  # if name is empty, set default name 'deepeye'
        else:
            self.name = name
        self.is_table_info = False
        self.import_method = methods_of_import[0]  # = none
        self.rank_method = methods_of_ranking[0]  # = none
    
    def table_info(self, name, column_info, *column_info2):
        """
        Input the table_info.

        Args:
            name(str): The name of the table(s)
            column_info(list): The name of each column
            column_info2(list or tuble or dict, usually list): The type of each column

        Returns:
            None
            
        """
        self.table_name = name
        self.column_names = []
        self.column_types = []
        self.column_dtypes = []
        if isinstance(column_info, list) and isinstance(column_info2[0], list):
            self.column_names = column_info
            self.column_types = column_info2[0]
            self.column_dtypes = column_info2[1]
        elif isinstance(column_info, dict):
            self.column_names = column_info.keys()
            self.column_types = column_info.values()
        elif isinstance(column_info, list) and isinstance(column_info[0], tuple):
            self.column_names = [i[0] for i in column_info]
            self.column_types = [i[1] for i in column_info]
        else:
            raise TypeError("unsupported argument types (%s, %s)" % (type(column_info), type(column_info2)))
        for idx, val in enumerate(self.column_types):
            if Type.getType(val.lower()) == 0:  # not a normal type
                raise Exception("doesnt support this column_type \' %s \' of column name \' %s \',please check Readme for specification " % (
                    val, self.column_names[idx]))
        self.is_table_info = True
    
    def error_throw(self, stage):
        """
        Find if there are errors at the beginning of each function in this file

        Args:
            stage(str): distinguish which function calls errow_throw:
                        rank: call errow_throw when executing the ranking function
                        output: call errow_throw when ececuting the output function

        Returns:
            None
            
        """
        if self.is_table_info == False:
            print("please enter table info by table_info()")
            sys.exit(0)
        if stage == 'rank':
            if self.import_method == 'none':
                self.error_output_import()
        elif stage == 'output':
            if self.import_method == 'none':
                self.error_output_import()
            else:
                if self.rank_method == 'none':
                    self.error_output_rank()
    
    def error_output_import(self):
        """
        Print import error information

        Args:
            None

        Returns:
            None
            
        """
        im_methods_string = ''
        for i in range(len(methods_of_import)):
            if i == 0:
                continue
            elif i != len(methods_of_import) - 1:
                im_methods_string += ('from_' + methods_of_import[i] + '() or ')
            else:  # i == len(methods_of_import)
                im_methods_string += ('from_' + methods_of_import[i] + '()')
        print("please import by " + im_methods_string)
        sys.exit(0)
    
    def error_output_rank(self):
        """
        Print rank error information

        Args:
            None

        Returns:
            None
            
        """
        rank_method_string = ''
        for i in range(len(methods_of_ranking)):
            if i == 0:
                continue
            elif i != len(methods_of_ranking) - 1:
                rank_method_string += (methods_of_ranking[i] + '() or ')
            else:
                rank_method_string += (methods_of_ranking[i] + '()')
        print("please rank first by " + rank_method_string)
        sys.exit(0)
    
    ##### data import function
    def from_csv(self, path, types=None):
        """
        read the csv file

        Args:
            path(str): the path of the csv file

        Returns:
            None
            
        """
        self.csv_path = path
        
        try:
            fh = open(self.csv_path, "r")
        except IOError:
            print("Error: no such file or directory")
        
        self.csv_dataframe = pd.read_csv(self.csv_path, header=0, keep_default_na=False, dtype=types).dropna(axis=0, how='any')
        test = self.csv_dataframe
        types = [0 for i in range(len(test.dtypes))]
        
        x = test.columns.values.tolist()
        global_variables.columns = test.columns.values.tolist()
        
        # type transformation
        for i in range(len(test.dtypes)):
            if test.dtypes[i].name[0:3] == 'int' or test.dtypes[i].name[0:5] == 'float':
                if (x[i][0] == "'" or x[i][0] == '"'):
                    x[i] = x[i].replace('\'', '').replace('"', '')
                for j in test[x[i]]:
                    if not (j == 0 or (j > 1000 and j < 2100)):
                        types[i] = test.dtypes[i].name[0:5]
                        break
                    else:
                        types[i] = 'year'
            elif test.dtypes[i].name[0:6] == 'object':
                if (x[i][0] == "'" or x[i][0] == '"'):
                    x[i] = x[i].replace('\'', '').replace('"', '')
                for j in test[x[i]]:
                    if j != 0 and not (re.search(r'\d+[/-]\d+[/-]\d+', j)):
                        types[i] = 'varchar'
                        break
                    else:
                        types[i] = 'date'
        
        name = path.rsplit('/', 1)[-1][:-4]
        self.table_info(name, x, types, test.dtypes)
        self.import_method = methods_of_import[2]  # = 'csv'
        
        # self.show_csv_info()
    
    def csv_handle(self, instance):
        """
        format the data according to the type

        Args:
            instance(Instance): the object of class Instance

        Returns:
            the instance object with the infomation(names, types, etc.)
            
        """
        table_origin = self.csv_dataframe
        in_column_num = len(self.column_names)
        in_column_name = self.column_names
        in_column_type = self.column_types
        in_column_dtype = self.column_dtypes
        
        instance.column_num = instance.tables[0].column_num = in_column_num
        for i in range(instance.column_num):
            instance.tables[0].names.append(in_column_name[i])
            instance.tables[0].types.append(Type.getType(in_column_type[i].lower()))  # getType is a static method
            instance.tables[0].dtypes.append(in_column_dtype[i])
        instance.tables[0].origins = [i for i in range(instance.tables[0].column_num)]
        
        instance.tuple_num = instance.tables[0].tuple_num = table_origin.shape[0]  # the number of rows
        for i in range(instance.tables[0].column_num):
            if instance.tables[0].types[i] == 3:  # if there is date type column in csv,convert into datetime format
                col_name = table_origin.columns[i]
                col_type = self.column_types[i]
                self.csv_handle_changedate(col_name, col_type)
        
        # change table column name with table_info column_names (for date type columns)
        for i in range(len(table_origin.columns)):
            table_origin.rename(columns={table_origin.columns[i]: in_column_name[i]}, inplace=True)
        instance.tables[0].D = table_origin.values.tolist()  # dataframe to list type and feed to D(where to store all the table info )
        return instance
    
    def csv_handle_changedate(self, col_name, col_type):
        """
        deal with date type data, wrap to datetime format

        Args:
            col_name(str): the name of columns
            col_type(str): the type of columns

        Returns:
            None
            
        """
        table = self.csv_dataframe
        if col_type == 'date':
            table[col_name] = pd.to_datetime(table[col_name]).dt.date
        elif col_type == 'datetime':
            table[col_name] = pd.to_datetime(table[col_name]).dt.to_pydatetime()
        elif col_type == 'year':
            table[col_name] = pd.to_datetime(table[col_name].apply(lambda x: str(int(x)) + '/1/1')).dt.date
    
    def show_csv_info(self):
        """
        print out csv info

        Args:

        Returns:
            None
            
        """
        print()
        display(HTML(self.csv_dataframe.head(10).to_html()))
    
    ##### ranking function
    def rank_generate_all_views(self, instance):
        """
        initialize before ranking 


        Args:
            instance(Instance): The object of class Instance

        Returns:
            instance with tables added
            
        """
        if len(instance.tables[0].D) == 0:
            print('no data in table')
            sys.exit(0)
        # print(instance.table_num, instance.view_num)
        with monitor('Solve', level=0):
            with monitor('Enumerate Tables', level=1):
                instance.addTables(instance.tables[0].dealWithTable())  # the first deal with is to transform the table into several small ones
                # print(instance.table_num, instance.view_num)
                begin_id = 1
                while begin_id < instance.table_num:
                    instance.tables[begin_id].dealWithTable()  # to generate views
                    begin_id += 1
                if instance.view_num == 0:
                    print('no chart generated')
                    sys.exit(0)
                # print(instance.table_num, instance.view_num)
                return instance
    
    def partial_order(self):
        """
        use partial order method to rank the charts

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('rank')
        instance = Instance(self.table_name)
        instance.addTable(Table(instance, False, '', ''))  # 'False'->transformed '',''->no describe yet
        if self.import_method == 'mysql':
            instance = self.mysql_handle(instance)
        elif self.import_method == 'csv':
            instance = self.csv_handle(instance)
        
        self.rank_partial(instance)
        
        self.rank_method = methods_of_ranking[2]  # = 'partial_order'
    
    def diversified_ranking(self):
        """
        Use diversified ranking method to rank the charts.

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('rank')
        instance = Instance(self.table_name)
        instance.addTable(Table(instance, False, '', ''))  # 'False'->transformed '',''->no describe yet
        if self.import_method == 'mysql':
            instance = self.mysql_handle(instance)
        elif self.import_method == 'csv':
            instance = self.csv_handle(instance)
        
        self.rank_partial(instance)
        
        self.rank_method = methods_of_ranking[3]  # = 'diversified_ranking'
    
    def rank_learning(self, instance):
        """
        inner function of learning_to_rank

        Args:
            instance(Instance): The object of class Instance.
            
        Returns:
            None
            
        """
        instance = self.rank_generate_all_views(instance)
        instance.getScore_learning_to_rank()
        self.instance = instance
    
    def rank_partial(self, instance):
        """
        inner function of partial_order and diversified_ranking

        Args:
            instance(Instance): The object of class Instance.
            
        Returns:
            None
            
        """
        instance = self.rank_generate_all_views(instance)
        instance.getM()
        instance.getW()
        instance.getScore()
        self.instance = instance
    
    ##### output function : list, print, single_json, multiple_jsons, single_html, multiple_htmls, 6 in total.
    def to_list(self):
        """
        export as list type

        Args:
            None
            
        Returns:
            the export list
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            export_list = self.output_div('list')
        else:
            export_list = self.output('list')
        return export_list
    
    def to_list_benchmark(self):
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
    
    def to_print_out(self):
        """
        print out to cmd

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('print')
        else:
            self.output('print')
    
    def to_single_json(self):
        """
        create a single json file

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('single_json')
        else:
            self.output('single_json')
    
    def to_multiple_jsons(self):
        """
        create multiple json files

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('multiple_jsons')
        else:
            self.output('multiple_jsons')
    
    def to_single_html(self):
        """
        convert to html by pyecharts and output to single html file

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('single_html')
        else:
            self.output('single_html')
    
    def to_multiple_htmls(self):
        """
        convert to html by pyecharts and output to multiple html files

        Args:
            None
            
        Returns:
            None
            
        """
        self.error_throw('output')
        
        if self.rank_method == methods_of_ranking[3]:  # 'diversified_ranking'
            self.output_div('multiple_htmls')
        else:
            self.output('multiple_htmls')
    
    def output_div(self, output_method):
        pass
    
    def output(self, output_method):
        """
        output function of partial_order and learning_to_rank for all kinds of output

        Args:
            output_method(str): output method:
                                list: to list
                                print: print to console
                                single_json/multiple_jsons: single/multiple json file(s)
                                single_html/multiple_htmls: single/multiple html file(s)
            
        Returns:
            None
            
        """
        instance = self.instance
        export_list = []
        order1 = order2 = 1
        old_view = ''
        if output_method == 'list':
            for i in range(instance.view_num):
                view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                if old_view:
                    order2 = 1
                    order1 += 1
                export_list.append(view.output(order1))
                old_view = view
            return export_list
        elif output_method == 'print':
            for i in range(instance.view_num):
                view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                if old_view:
                    order2 = 1
                    order1 += 1
                pprint(view.output(order1))
                old_view = view
            return
        elif output_method == 'single_json' or output_method == 'multiple_jsons':
            path2 = os.getcwd() + '/json/'
            if not os.path.exists(path2):
                os.mkdir(path2)
            if output_method == 'single_json':
                f = open(path2 + self.table_name + '.json', 'w')
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    f.write(view.output(order1) + '\n')
                    old_view = view
                f.close()  # Notice that f.close() is out of the loop to create only one file
            else:  # if output_method == 'multiple_jsons'
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    f = open(path2 + self.table_name + str(order1) + '.json', 'w')
                    f.write(view.output(order1))
                    f.close()  # Notice that f.close() is in the loop to create multiple files
                    old_view = view
            return
        elif output_method == 'single_html' or output_method == 'multiple_htmls':
            path2 = os.getcwd() + '/html/'
            if not os.path.exists(path2):
                os.mkdir(path2)
            page = Page()
            if output_method == 'single_html':
                self.page = Page()
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    old_view = view
                    self.html_output(order1, view, 'single')
                self.page.render('./html/' + self.table_name + '_all' + '.html')
            else:  # if output_method == 'multiple_htmls'
                path3 = os.getcwd() + '/html/' + self.table_name
                if not os.path.exists(path3):
                    os.mkdir(path3)
                for i in range(instance.view_num):
                    view = instance.tables[instance.views[i].table_pos].views[instance.views[i].view_pos]
                    if old_view:
                        order2 = 1
                        order1 += 1
                    old_view = view
                    self.html_output(order1, view, 'multiple')
            return
    
    def html_output(self, order, view, mode):
        """
        output function of html

        Args:
            order(int): diversified_ranking use different order
            view(View): view object
            mode(str): single or multiple
            
        Returns:
            None
            
        """
        instance = self.instance
        data = {}
        data['order'] = order
        data['chartname'] = instance.table_name
        data['describe'] = view.table.describe
        data['x_name'] = view.fx.name
        data['y_name'] = view.fy.name
        data['chart'] = Chart.chart[view.chart]
        data['classify'] = [v[0] for v in view.table.classes]
        data['x_data'] = view.X
        data['y_data'] = view.Y
        data['title_top'] = 5
        
        [chart, filename] = self.html_handle(data)
        grid = Grid()
        grid.add(chart, grid_opts=opts.GridOpts(pos_bottom='20%', pos_top='20%'))
        if mode == 'single':
            self.page.add(grid)  # the grid is added in the same page
        elif mode == 'multiple':
            grid.render('./html/' + self.table_name + '/' + filename)  # the grid is added in a new file
    
    def html_handle(self, data):
        """
        convert function to html by pyecharts

        Args:
            data(dict): the data info
            
        Returns:
            chart: chart generated by pyecharts: Bar, Pie, Line or Scatter
            filename: html file name
            
        """
        
        filename = self.table_name + str(data['order']) + '.html'
        margin = str(data['title_top']) + '%'
        # 设置图标基本属性
        if data['chart'] == 'bar':
            chart = (Bar().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
                xaxis_opts=opts.AxisOpts(name=data['x_name']),
                yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
        elif data['chart'] == 'pie':
            chart = (Pie().set_global_opts(
                title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin)))
        elif data['chart'] == 'line':
            chart = (Line().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
                xaxis_opts=opts.AxisOpts(name=data['x_name']),
                yaxis_opts=opts.AxisOpts(name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
        elif data['chart'] == 'scatter':
            chart = (Scatter().set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                title_opts=opts.TitleOpts(title=data['chartname'], subtitle=data['describe'], pos_left='center', pos_top=margin),
                xaxis_opts=opts.AxisOpts(type_='value', name=data['x_name'], splitline_opts=opts.SplitLineOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(type_='value', name=data['y_name'], splitline_opts=opts.SplitLineOpts(is_show=True))))
        else:
            print("not valid chart")
        
        if not data["classify"]:  # 在图片上只需展示一组数据
            attr = data["x_data"][0]  # 横坐标
            val = data["y_data"][0]  # 纵坐标
            if data['chart'] == 'bar':
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
            elif data['chart'] == 'line':
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
            elif data['chart'] == 'pie':
                chart.add("", [list(z) for z in zip(attr, val)])
            elif data['chart'] == 'scatter':
                if isinstance(attr[0], str):
                    attr = [x for x in attr if x != '']
                    attr = list(map(float, attr))
                if isinstance(val[0], str):
                    val = [x for x in val if x != '']
                    val = list(map(float, val))
                chart.add_xaxis(attr).add_yaxis("", val, label_opts=opts.LabelOpts(is_show=False))
        else:  # 在图片上需要展示多组数据
            attr = data["x_data"][0]  # 横坐标
            for i in range(len(data["classify"])):  # 循环输出每组数据
                val = data["y_data"][i]  # 每组纵坐标的值
                name = (data["classify"][i][0] if type(data["classify"][i]) == type(('a', 'b')) else data["classify"][i])
                if i == 0:
                    if data['chart'] != 'pie' and data['chart'] != 'scatter':
                        chart.add_xaxis(attr)
                if data['chart'] == 'bar':
                    chart.add_yaxis(name, val, stack="stack1", label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'line':
                    chart.add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
                elif data['chart'] == 'pie':
                    chart.add("", [list(z) for z in zip(attr, val)])
                elif data['chart'] == 'scatter':
                    attr_scatter = data["x_data"][i]
                    if isinstance(attr_scatter[0], str):  # 去除散点图的空点，并将字符类型转化为数字类型
                        attr_scatter = [x for x in attr_scatter if x != '']
                        attr_scatter = list(map(float, attr_scatter))
                    if isinstance(val[0], str):
                        val = [x for x in val if x != '']
                        val = list(map(float, val))
                    chart.add_xaxis(attr_scatter).add_yaxis(name, val, label_opts=opts.LabelOpts(is_show=False))
        return chart, filename
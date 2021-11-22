import math

from scipy import stats
from VizGRank.dp_pack import Chart
from VizGRank.dp_pack.view import View


def compisite_order(view: View):
    return view.table.instance.view_num - view.view_id


def matching_quality(view: View):
    return view.M


def chart_personalization(view: View):
    corr = 0
    if view.chart == Chart.pie:
        corr = stats.entropy(view.Y[0])
    elif view.chart == Chart.bar:
        if view.series_num == 1:
            corr = stats.entropy(view.Y[0])
        else:
            f, p = stats.f_oneway(*view.Y)
            corr = 1 - p
    elif view.chart == Chart.scatter or view.chart == Chart.line:
        if view.series_num == 1:
            corr = view.getCorrelation(0)
        else:
            corr = max([view.getCorrelation(i) for i in range(view.series_num)])
    
    if math.isnan(corr) or math.isinf(corr):
        corr = 0
    
    return corr

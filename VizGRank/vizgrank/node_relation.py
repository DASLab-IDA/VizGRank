from VizGRank.dp_pack import Chart


def calc_similarity_tuple(view_i, view_j):
    viz_a = []
    viz_b = []
    # get all columns
    columns_a = [(view_i.y_name_origin, view_i.trans_y), (view_i.x_name_origin, view_i.trans_x)]
    if view_i.z_id != -1:
        columns_a.append((view_i.z_name_origin, view_i.trans_z))
    columns_b = [(view_j.y_name_origin, view_j.trans_y), (view_j.x_name_origin, view_j.trans_x)]
    if view_j.z_id != -1:
        columns_b.append((view_j.z_name_origin, view_j.trans_z))
    viz_a.extend(columns_a)
    viz_b.extend(columns_b)
    # get chart type
    viz_a.append(Chart.chart[view_i.chart])
    viz_b.append(Chart.chart[view_j.chart])
    set1, set2 = set(viz_a), set(viz_b)
    similarity = len(set1 & set2) / float(len(set1 | set2))
    return similarity


def calc_similarity_set(view_i, view_j):
    viz_a = []
    viz_b = []
    # get all columns
    columns_a = [view_i.y_name_origin, view_i.trans_y, view_i.x_name_origin, view_i.trans_x]
    if view_i.z_id != -1:
        columns_a.extend([view_i.z_name_origin, view_i.trans_z])
    columns_b = [view_j.y_name_origin, view_j.trans_y, view_j.x_name_origin, view_j.trans_x]
    if view_j.z_id != -1:
        columns_b.extend([view_j.z_name_origin, view_j.trans_z])
    viz_a.extend(columns_a)
    viz_b.extend(columns_b)
    # get chart type
    viz_a.append(Chart.chart[view_i.chart])
    viz_b.append(Chart.chart[view_j.chart])
    set1, set2 = set(viz_a), set(viz_b)
    similarity = len(set1 & set2) / float(len(set1 | set2))
    return similarity


calc_similarity = calc_similarity_set


def context_similarity(view_i, view_j, adjacent_only=False):
    similarity = calc_similarity(view_i, view_j)
    
    if adjacent_only:
        similarity = 1 if similarity > 0 else 0
    
    return similarity


def context_dissimilarity(view_i, view_j, adjacent_only=False):
    similarity = calc_similarity(view_i, view_j)
    
    if adjacent_only:
        similarity = 1 if similarity > 0 else 0
    
    return 1 - similarity

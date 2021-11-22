import json
import sys

sys.path.append('..')
# import the VizGRank API
from VizGRank.vizgrank.vizgrank import VizGRank

# create a VizGRank instance
vr = VizGRank('demo')

# the csv file to process
csv_path = 'example/data.csv'

# you can specify the data type of each columns manually in the file 'types.json'
# or just leave it to the pandas library (by assigning None to types).
dtypes = None
type_path = 'example/types.json'
with open(type_path) as f:
    types = json.load(f)

# Followings are the steps to generated VizGRank results:
# First, you read the data into VizGRank instance by invoking read_day.
# Second you generate visualization candidates by invoking generate_visualizations.
# Then you rank visualizations by invoking rank_visualizations.
# Finally you output a list of visualizations with data by invoking output_visualizations.
r = vr.read_data(csv_path, dtypes=dtypes) \
    .generate_visualizations() \
    .rank_visualizations() \
    .output_visualizations()

# There is an alternative to generate a html file with drawn visualizations on canvas for demonstration.
vr.to_html()

import json
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import mplleaflet
import time
# Load up the geojson data
filename = os.path.join(os.path.dirname(__file__), 'data', '/Users/pelanmar1/Coding/Tesis/heuristics/cluster0.geojson')
with open(filename) as f:
    gj = json.load(f)


# Grab the coordinates (longitude, latitude) from the features, which we
# know are Points
xy = np.array([feat['geometry']['coordinates'] for feat in gj['features'][::]])
# Plot the path as red dots connected by a blue line

# plt.hold(True)
plt.plot(xy[1:,0], xy[1:,1], 'r.', markersize=25)
plt.plot(xy[0,0], xy[0,1], 'g.', markersize=30)

def draw_line_between(node_a,node_b,plt,alpha=1, linewidth=1):
    node_x = [xy[node_a,0],xy[node_b,0]]
    node_y = [xy[node_a,1],xy[node_b,1]]
    plt.plot(node_x,node_y,'b',alpha=alpha, linewidth=linewidth)

def tour_to_xy(tour,plt,alpha=1, linewidth=1):
    n = len(tour)
    for i in range(n-1):
        a = tour[i]
        b = tour[i+1]
        draw_line_between(a,b,plt,alpha=alpha,linewidth=linewidth)
    draw_line_between(tour[-1],tour[0],plt)

def draw_all_connection(xy,plt,alpha=0.1,linewidth=1):
    for i in range(len(xy)):
        for j in range(len(xy)):
            draw_line_between(i,j,plt,alpha=alpha,linewidth=linewidth)

        
tour = [17, 4, 20, 5, 2, 9, 10, 13, 8, 19, 11, 23, 16, 21, 6, 14, 0, 18, 22, 7, 15, 1, 12, 3, 17]
tour_to_xy(tour,plt,linewidth=5)
#for i in range(100):
draw_all_connection(xy,plt,alpha=1)

# ax =plt.axes()
# plt.text(0.5, 0.5,'matplotlib',
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes, fontsize=100)

root, ext = os.path.splitext(__file__)
mapfile = root  + 'test.html'
# Create the map. Save the file to basic_plot.html. _map.html is the default
# if 'path' is not specified
mplleaflet.show(path=mapfile)
time.sleep(3)
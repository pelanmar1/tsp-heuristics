import networkx as nx
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

class TSPPlotter:
    def save_run_results(self, file_name, data):
        with open(file_name,"wb") as filehandle:
            pickle.dump(data,filehandle)

    def csv_2_tuple_list(self,csv_fn):
        df = pd.read_csv(csv_fn)
        return list(zip(df.iloc[:,0],df.iloc[:,1]))

    def cost_plot(self,run_data, title="Graph", xlab="x", ylab="y"):
        x = []
        y = []
        for idx,elem in enumerate(run_data):
            x.append(idx)
            y.append(elem["tour_length"])
        plt.plot(x,y,"b")
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()

    def plot_coords(self,coords, graph, heavy_nodes=None):
        G=nx.Graph()
        
        # Set coordinates
        for idx,c in enumerate(coords):
            G.add_node(idx,pos=(c[1],c[0]))
        # Set weights
        for i in range(len(coords)):
            for j in range(len(coords)):
                G.add_edge(i,j, weight=graph[i][j])

        pos=nx.get_node_attributes(G,'pos')
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='r')

        if heavy_nodes is not None:
            heavy_edges = []
            light_edges = []
            for (u,v,_) in G.edges(data=True):
                if (u,v) in heavy_nodes:
                    heavy_edges.append((u,v))
                else:
                    light_edges.append((u,v))
            nx.draw_networkx_edges(G, pos, edgelist=heavy_edges,width=3, alpha=1, edge_color="b")
            nx.draw_networkx_edges(G, pos, edgelist=light_edges,width=1, alpha=0.1, edge_color='b', style='solid')
        else:
            nx.draw_networkx_edges(G, pos,width=1, alpha=0.1, edge_color="b")
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        plt.show()
        return plt


    def plot_coords_dyn(self,coords, graph, heavy_nodes=None):
        G=nx.Graph()
        
        # Set coordinates
        for idx,c in enumerate(coords):
            G.add_node(idx,pos=(c[1],c[0]))
        pos=nx.get_node_attributes(G,'pos')
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='r')

        if heavy_nodes is not None:
            heavy_edges = []
            light_edges = []
            for (u,v,_) in G.edges(data=True):
                if (u,v) in heavy_nodes:
                    heavy_edges.append((u,v))
                else:
                    light_edges.append((u,v))
            nx.draw_networkx_edges(G, pos, edgelist=heavy_edges,width=3, alpha=1, edge_color="b")
            nx.draw_networkx_edges(G, pos, edgelist=light_edges,width=1, alpha=0.1, edge_color='b', style='solid')
        else:
            nx.draw_networkx_edges(G, pos,width=1, alpha=0.1, edge_color="b")
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        plt.show()
        return plt


    def plot(self,graph):
        A = np.matrix(graph)
        G = nx.from_numpy_matrix(graph)
        pos = nx.spring_layout(G)
        edge_labels = dict([((u, v,), d['weight'])
                            for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        nx.draw(G, pos, with_labels = True)
        plt.show()

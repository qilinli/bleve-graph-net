import torch
from torch_geometric.data import Data
import enum

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

class NodeType_BLEVE(enum.IntEnum):
    NORMAL = 0
    TANK = 1
    GROUND = 2



# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, y, pos, next_y, next2_y, edge_attr = None, None, None, None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="y":
            y = graph.y
        elif key=="pos":
            pos = graph.pos
        elif key=="next_y":
            next_y = graph.next_y
        elif key=="next2_y":
            next2_y = graph.next2_y
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        else:
            pass
    return (x, edge_index, y, pos, next_y, next2_y, edge_attr)

# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    x, edge_index, y, pos, next_y, next2_y, edge_attr = decompose_graph(graph)
    
    g = Data(x=x, edge_index=edge_index, y=y, pos=pos, next_y=next_y, next2_y=next2_y, edge_attr=edge_attr)
    
    return g

def update_graph(graph, pred_p):
    """return a updated graph where x is updated with the predicted p, and y is the next p.
    """
    x, edge_index, y, pos, next_y, next2_y, edge_attr = decompose_graph(graph)
    node_attr = torch.cat((pred_p.reshape(-1,1), x[:, 1:]), dim=1)
    g = Data(x=node_attr, edge_index=edge_index, y=next_y, pos=pos, next_y=next2_y, next2_y=next2_y, edge_attr=edge_attr)
    
    return g

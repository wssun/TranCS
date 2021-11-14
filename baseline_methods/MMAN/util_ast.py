import numpy as np
import argparse
import json
from collections import Counter
import networkx as nx
import dgl

PAD_ID, UNK_ID, EOS_ID = [0, 1, 2]


def build_tree(tree_json, ast_dict):
    g = nx.DiGraph()

    def _rec_build(nid, idx, t_json):
        children = [c for c in t_json[idx]['children']]
        idx_str = idx.split('1*NODEFIX')[1]

        if len(children) == 2:

            if nid is None:
                g.add_node(0, x=-1, y=int(idx_str), mask=0)
                nid = 0

            for c in children:
                cid = g.number_of_nodes()

                y_value_int = int(c.split('1*NODEFIX')[1])

                c_children = t_json[c]['children']

                if len(c_children) == 2:
                    g.add_node(cid, x=-1, y=y_value_int, mask=0)
                    _rec_build(cid, c, t_json)

                else:
                    assert len(t_json[c]['children']) == 1
                    word_index = ast_dict.get(t_json[c]['children'][0], UNK_ID)
                    g.add_node(cid, x=word_index, y=y_value_int, mask=1)

                g.add_edge(cid, nid)

        else:
            assert len(t_json[idx]['children']) == 1
            word_index = ast_dict.get(t_json[idx]['children'][0], UNK_ID)
            if nid is None:
                cid = 0
            else:
                cid = g.number_of_nodes()

            g.add_node(cid, x=word_index, y=int(idx), mask=1)

            if nid is not None:
                g.add_edge(cid, nid)

    for k, node in tree_json.items():
        if node['parent'] == None:
            root_idx = k
            break

    _rec_build(None, root_idx, tree_json)
    ret = dgl.DGLGraph()
    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])

    return ret
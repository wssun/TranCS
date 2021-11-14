import numpy as np

def get_one_cfg_npy_info(json_cfg_dict, n_node, n_edge_types, state_dim, max_word_num):
    node_num = min(len(json_cfg_dict), n_node)
    save_node_feature_dict, save_edge_digit_list = {}, []
    anno = np.zeros([n_node, max_word_num])

    node_index = 0
    i = 0
    while node_index < node_num:
        if str(i) not in json_cfg_dict.keys():
            i += 1
            continue

        word_list = json_cfg_dict[str(i)]['wordid']
        word_num_this_node = len(word_list)

        for j in range(0, min(max_word_num, word_num_this_node)):
            anno[i][j] = word_list[j]

        if 'snode' in json_cfg_dict[str(i)].keys():
            snode_list = json_cfg_dict[str(i)]['snode']
            for k in range(0, min(n_node, len(snode_list))):
                snode = snode_list[k]
                if snode < n_node:
                    save_edge_digit_list.append([i, snode, 0])

        i += 1
        node_index += 1

    adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
    node_mask = [1 if k < node_num else 0 for k in range(0, n_node)]
    init_input = pad_one_anno(anno, n_node, state_dim, max_word_num)

    return adjmat, init_input, node_mask

def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        tgt_idx = edge[1]
        e_type = edge[2]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a

def pad_one_anno(anno, n_node, state_dim, annotation_dim):
    padding = np.zeros([n_node, (state_dim - annotation_dim)])
    init_input = np.concatenate((anno, padding), 1)
    return init_input


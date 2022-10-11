import torch
import torch.nn as nn
from math import floor

from utils.pareto_utils import *


class Node:
    def __init__(self, cost=0, BOPs=0, profit=0, bit=None, bit_a=None, parent=None, left_4=None, left_8=None, middle_4=None, middle_8=None, right_4=None, right_8=None, position='middle'):
        self.parent = parent
        self.left_4 = left_4            # bit-precision: [2,4]
        self.left_8 = left_8            # bit-precision: [2,8]
        self.middle_4 = middle_4        # bit-precision: [4,4]
        self.middle_8 = middle_8        # bit-precision: [4,8]
        self.right_4 = right_4          # bit-precision: [8,4]
        self.right_8 = right_8          # bit-precision: [8,8]
        self.position = position
        self.cost = cost
        self.BOPs = BOPs
        self.profit = profit
        self.bit = bit
        self.bit_a = bit_a
    def __str__(self):
        return 'cost: {:.2f} profit: {:.2f}'.format(self.cost, self.profit)
    def __repr__(self):
        return self.__str__()


def pareto_frontier(model, metric, paras, MACs):
    layers = []
    names = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            names.append(name)
            layers.append(layer.weight)

    sen_result = [[0 for i in range(len(layers))] for j in range(3)]
    for i in range(len(layers)):
        for j, num_bits in enumerate([2, 4, 8]):
            sen_result[j][i] = compute_perturbation(layers[i], num_bits, metric[i])

    sen_a_result = [[0 for i in range(len(layers))] for j in range(2)]
    for i in range(len(layers)):
        for j, num_bits in enumerate([4, 8]):
            sen_a_result[j][i] = compute_perturbation(layers[i], num_bits, metric[i])

    # bit-precision options
    bits = [2, 4, 8]
    bits_a = [4, 8]

    # initialize root node
    prifits = sen_result
    prifits_a = sen_a_result
    root = Node(cost=0, profit=0, parent=None)
    current_list = [root]

    # 3D pareto frontier
    for layer_id in range(len(layers)):
        # 1. new leaf nodes
        next_list = []
        print('\t Loop', layer_id, 'th layer ...')
        if 'downsample' not in names[layer_id]:
            for n in current_list:
                n.left_4 = Node(n.cost + bits[0] * paras[layer_id] / 8, n.BOPs + bits[0] * bits_a[0] * MACs[layer_id],
                                n.profit + prifits[0][layer_id] + prifits_a[0][layer_id], bit=bits[0], bit_a=bits_a[0], parent=n, position='left_4')
                n.left_8 = Node(n.cost + bits[0] * paras[layer_id] / 8, n.BOPs + bits[0] * bits_a[1] * MACs[layer_id],
                                n.profit + prifits[0][layer_id] + prifits_a[1][layer_id], bit=bits[0], bit_a=bits_a[1], parent=n, position='left_8')
                n.middle_4 = Node(n.cost + bits[1] * paras[layer_id] / 8, n.BOPs + bits[1] * bits_a[0] * MACs[layer_id],
                                  n.profit + prifits[1][layer_id] + prifits_a[0][layer_id], bit=bits[1], bit_a=bits_a[0], parent=n, position='middle_4')
                n.middle_8 = Node(n.cost + bits[1] * paras[layer_id] / 8, n.BOPs + bits[1] * bits_a[1] * MACs[layer_id],
                                  n.profit + prifits[1][layer_id] + prifits_a[1][layer_id], bit=bits[1], bit_a=bits_a[1], parent=n, position='middle_8')
                n.right_4 = Node(n.cost + bits[2] * paras[layer_id] / 8, n.BOPs + bits[2] * bits_a[0] * MACs[layer_id],
                                 n.profit + prifits[2][layer_id] + prifits_a[0][layer_id], bit=bits[2], bit_a=bits_a[0], parent=n, position='right_4')
                n.right_8 = Node(n.cost + bits[2] * paras[layer_id] / 8, n.BOPs + bits[2] * bits_a[1] * MACs[layer_id],
                                 n.profit + prifits[2][layer_id] + prifits_a[1][layer_id], bit=bits[2], bit_a=bits_a[1], parent=n, position='right_8')
                next_list.extend([n.left_4, n.left_8, n.middle_4, n.middle_8, n.right_4, n.right_8])
        else:
            for n in current_list:
                n.left_d = Node(n.cost + bits[0] * paras[layer_id] / 8, n.BOPs + bits[0] * MACs[layer_id],
                                n.profit + prifits[0][layer_id], bit=bits[0], bit_a=0, parent=n, position='left_d')
                n.middle_d = Node(n.cost + bits[1] * paras[layer_id] / 8, n.BOPs + bits[1] * MACs[layer_id],
                                  n.profit + prifits[1][layer_id], bit=bits[1], bit_a=0, parent=n, position='middle_d')
                n.right_d = Node(n.cost + bits[2] * paras[layer_id] / 8, n.BOPs + bits[2] * MACs[layer_id],
                                 n.profit + prifits[2][layer_id], bit=bits[2], bit_a=0, parent=n, position='right_d')
                next_list.extend([n.left_d, n.middle_d, n.right_d])

        # 2. sort and select -- model size
        pruned_list = []
        next_list_r = next_list
        next_list.sort(key=lambda x: x.cost, reverse=False)
        for node in next_list:
            if len(pruned_list) == 0 or node.profit <= pruned_list[-floor(1/20*len(pruned_list))].profit:
                pruned_list.append(node)
            else:
                node.parent.__dict__[node.position] = None

        # 3. sort and select -- BitOps
        pruned_list_r = []
        next_list_r.sort(key=lambda x: x.BOPs, reverse=False)
        for node in next_list_r:
            if len(pruned_list_r) == 0 or node.profit <= pruned_list_r[-floor(1/20*len(pruned_list_r))].profit:
                pruned_list_r.append(node)
            else:
                node.parent.__dict__[node.position] = None
                
        # 4. loop
        current_list = list(set(pruned_list + pruned_list_r))

    sizes = [x.cost for x in current_list]
    BOPs = [x.BOPs for x in current_list]
    sens = [x.profit for x in current_list]

    return sizes, BOPs, sens

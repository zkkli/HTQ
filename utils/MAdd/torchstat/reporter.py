import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)


def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)


def report_format(collected_nodes):
    MACs = []
    for node in collected_nodes:
        name = node.name
        print(name)
        if 'conv' in name or 'fc' in name:
            MAdd = node.MAdd
            MACs.append(MAdd)

    return MACs



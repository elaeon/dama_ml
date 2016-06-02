from operator import itemgetter

def order_2d(list_2d, index=(0, 1), block_size=60):
    """
    list_ = [(10, 50), (13, 100), (14, 40), (15, 90), (21, 30), (40, 10), (60, 20)]
    {block0: elems, block1: elems, ...}
    """
    blocks = build_blocks(list_2d, block_size, index)
    for row, items in blocks.items():
        if len(items) > 1:
            items.sort(key=itemgetter(index[1]))
    return blocks

def build_blocks(list_2d, block_size, index):
    """ build a dict of rows where each row is created if elem in 0 is grater 
        than block_size. Each row contains coords of numbers in the plane.
        list_ = [(10, 50), (13, 100), (14, 40), (15, 90), (21, 30), (40, 10), (60, 20)]
    """    
    g = itemgetter(index)
    data = sorted(list_2d, key=itemgetter(index[0]))
    initial = data.pop(0)
    blocks = {0: [initial]}
    block = 0
    while len(data) > 0:
        elem = data.pop(0)
        in_block = abs(initial[index[0]] - elem[index[0]]) <= block_size
        if not in_block:
            block += 1
        blocks.setdefault(block, [])
        blocks[block].append(elem)
        initial = elem
    return blocks

def order_table_print(headers, table, order_column):
    from tabulate import tabulate
    headers_lower = [h.lower() for h in headers]
    order_index = headers_lower.index(order_column)
    table = sorted(table, key=lambda x: x[order_index], reverse=True)
    print(tabulate(table, headers))

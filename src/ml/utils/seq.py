def grouper_chunk_filler(n, iterable, fillvalue=None):
    from itertools import izip_longest
    "grouper_chunk_filler(3, '[1,2,3,4,5,6,7]', '-') --> [1,2,3] [4,5,6] [7,-,-]"
    if not hasattr(object, '__iter__'):
        args = [iter(iterable)] * n
    else:
        args = [iterable] * n
    return izip_longest(fillvalue=fillvalue, *args)


def grouper_chunk(n, iterable):
    from itertools import islice, chain
    "grouper_chunk_filler(3, '[1,2,3,4,5,6,7]', '-') --> [1,2,3] [4,5,6] [7]"
    if not hasattr(object, '__iter__'):
        it = iter(iterable)
    else:
        it = iterable
    while True:
        chunk = islice(it, n)
        try:
            first_el = next(chunk)
        except StopIteration:
            return
        yield chain((first_el,), chunk)

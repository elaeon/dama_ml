from itertools import islice, chain


def grouper_chunk(n, iterable):
    "grouper_chunk(3, '[1,2,3,4,5,6,7]') --> [1,2,3] [4,5,6] [7]"
    it = iter(iterable)
    while True:
        chunk = islice(it, n)
        try:
            first_el = next(chunk)
        except StopIteration:
            return
        yield chain((first_el,), chunk)

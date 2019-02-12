def cache(func):
    def fn_wrapper(self):
        attr = "{}_cache".format(func.__name__)
        if not hasattr(self, attr) or getattr(self, attr) is None:
            setattr(self, attr, func(self))
        return getattr(self, attr)
    return fn_wrapper


def clean_cache(func):
    def fn_wrapper(self, value):
        attr = "{}_cache".format(func.__name__)
        if hasattr(self, attr) and getattr(self, attr) is not None:
            setattr(self, attr, None)
        return func(self, value)
    return fn_wrapper


def cut(fn):
    def check_params(*args, **kwargs):
        smx, end, length = fn(*args, **kwargs)
        if end < length:
            return smx[:end]
        return smx
    return check_params

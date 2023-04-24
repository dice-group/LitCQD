
class DatasetProperty(object):
    def __init__(self, attr):
        self.attr = "_" + attr

    def __get__(self, obj, objtype):
        if getattr(obj, self.attr) is None:
            raise Exception(f"DatasetProperty '{self.attr[1:]}' is accessed before it is loaded.")
        return getattr(obj, self.attr)

    def __set__(self, obj, value):
        setattr(obj, self.attr, value)

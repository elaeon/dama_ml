from sklearn.model_selection import train_test_split
from ml.data.groups.core import DaGroup


class CV(object):
    def __init__(self, group_data, group_target: str = None, train_size: float = .7,
                 valid_size: float = .1, unbalanced=None):
        self.train_size = train_size
        self.valid_size = valid_size
        self.unbalanced = unbalanced
        self.group_target = group_target
        self.group_data = group_data

    def apply(self, data) -> DaGroup:
        train_size = round(self.train_size + self.valid_size, 2)
        if self.group_target is not None:
            x_train, x_test, y_train, y_test = train_test_split(
                data[self.group_data], data[self.group_target], train_size=train_size, random_state=0)
            size = len(data)
            valid_size_index = int(round(size * self.valid_size, 0))
            x_validation = x_train[:valid_size_index]
            y_validation = y_train[:valid_size_index]
            x_train = x_train[valid_size_index:]
            y_train = y_train[valid_size_index:]
            if self.unbalanced is not None:
                return NotImplemented
            else:
                x_train.rename_group(self.group_data, "train_x")
                y_train.rename_group(self.group_target, "train_y")
                x_test.rename_group(self.group_data, "test_x")
                y_test.rename_group(self.group_target, "test_y")
                x_validation.rename_group(self.group_data, "validation_x")
                y_validation.rename_group(self.group_target, "validation_y")
                stc = x_train + y_train + x_test + y_test + x_validation + y_validation
                return stc
        else:
            x_train, x_test = train_test_split(data[self.group_data], train_size=train_size, random_state=0)
            size = len(data)
            valid_size_index = int(round(size * self.valid_size, 0))
            x_validation = x_train[:valid_size_index]
            x_train = x_train[valid_size_index:]

            if self.unbalanced is not None:
                return NotImplemented
            else:
                x_train.rename_group(self.group_data, "train_x")
                x_test.rename_group(self.group_data, "test_x")
                x_validation.rename_group(self.group_data, "validation_x")
                stc = x_train + x_test + x_validation
                return stc
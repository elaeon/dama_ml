from peewee import *

db = SqliteDatabase('ml.db')

class MlModel(Model):
    name = CharField()
    dataset = CharField()
    version = CharField()

    class Meta:
        database = db


class ModelEvaluation(Model):
    accuracy = FloatField()
    precision = FloatField()
    recall = FloatField()
    f1 = FloatField()
    logloss = FloatField()
    owner = ForeignKeyField(MlModel, related_name='evaluation')

    class Meta:
        database = db

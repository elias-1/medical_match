from django.db import models


class Property(models.Model):
    pid = models.AutoField(primary_key=True)
    entity_id = models.CharField(
        null=False, blank=False, max_length=20, db_index=True, unique=False)
    entity_type = models.CharField(
        null=False, blank=False, max_length=50, db_index=False, unique=False)
    property_name = models.CharField(
        null=False, blank=False, max_length=50, db_index=False, unique=False)
    property_value = models.CharField(
        null=False, blank=True, max_length=2000, db_index=False, unique=False)


class Entity_relation(models.Model):
    rid = models.AutoField(primary_key=True)
    entity_id1 = models.CharField(
        null=False, blank=False, max_length=20, db_index=True, unique=False)
    entity_name1 = models.CharField(
        null=False, blank=False, max_length=255, db_index=False, unique=False)
    entity_type1 = models.CharField(
        null=False, blank=False, max_length=50, db_index=False, unique=False)
    relation = models.CharField(
        null=False, blank=True, max_length=50, db_index=False, unique=False)
    entity_id2 = models.CharField(
        null=False, blank=False, max_length=20, db_index=True, unique=False)
    entity_name2 = models.CharField(
        null=False, blank=False, max_length=255, db_index=False, unique=False)
    entity_type2 = models.CharField(
        null=False, blank=False, max_length=50, db_index=False, unique=False)

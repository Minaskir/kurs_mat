from django.db import models

class TableData(models.Model):
    name = models.CharField(max_length=100)
    x = models.JSONField()
    y = models.JSONField()

    def __str__(self):
        return self.name
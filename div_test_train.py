import pandas as pd
import csv
from langdetect import detect

# read data
f = open("sentiment_analysis_train.v1.0.txt", "r", encoding="utf8")


def div():
    with open('test.csv', mode='w', encoding="utf8") as csv_file:
        fieldnames = ['label', 'content', 'lang']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        i = 0
        for x in f:
            if i > 15600:
                a = x.split(" ", 1)
                writer.writerow({'label': a[0], 'content': a[1].strip()})
            i = i + 1


div()

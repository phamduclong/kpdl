import pandas as pd
import csv

f = open("sentiment_analysis_test.v1.0.txt", "r",encoding="utf8")

with open('test_p.csv', mode='w',encoding="utf8") as csv_file:
    fieldnames = ['content']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for x in f:
        writer.writerow({'content': x.strip()})

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import csv
from   langdetect import  detect
# read data
# f = open("sentiment_analysis_train.v1.0.txt", "r",encoding="utf8")
#
# with open('data.csv', mode='w',encoding="utf8") as csv_file:
#     fieldnames = ['label', 'content','lang']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for x in f:
#         a = x.split(" ", 1)
#         writer.writerow({'label':a[0],'content': a[1].strip(),'lang':detect(a[1].strip())})


# clean data
reviews_df = pd.read_csv("data.csv")
langs=set(reviews_df["lang"])
for item in langs:
    df = pd.DataFrame(reviews_df[reviews_df.lang == item])
    df.to_csv('../data/'+item+'.csv',index=False,encoding="utf8",header=True)

    #print(reviews_df[reviews_df.lang == item])
    # with open(item+'.csv', mode='w',encoding="utf8") as csv_file:
    #     fieldnames = ['label', 'content','lang']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #         writer.writerow({'label':a[0],'content': a[1].strip(),'lang':a[2].strip()})










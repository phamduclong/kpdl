#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import csv
from   langdetect import  detect
from googletrans import Translator
# clean data
reviews_df = pd.read_csv("data.csv")
langs=set(reviews_df["lang"])
for item in langs:
    # df = pd.DataFrame(reviews_df[reviews_df.lang == item])
    # df.to_csv('../data/'+item+'.csv',index=False,encoding="utf8",header=True)

    #print(reviews_df[reviews_df.lang == item])
    # with open(item+'.csv', mode='w',encoding="utf8") as csv_file:
    #     fieldnames = ['label', 'content','lang']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #         writer.writerow({'label':a[0],'content': a[1].strip(),'lang':a[2].strip()})
    if item=='af' :
        #lang_data = pd.read_csv('../translate/'+item+'.csv')
        lang_data = open('../data/'+item+'.csv', "r", encoding="utf8")
        for x in lang_data:
            t= x.split(',')
            translator = Translator()
            a=translator.translate(t[1],dest='af')
            print ( a.text)









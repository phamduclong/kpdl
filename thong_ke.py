import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

reviews_df = pd.read_csv("data.csv")
label_1=reviews_df[reviews_df["label"]=='__label__tot']
print (len(label_1))
xuat_xac =len(reviews_df[reviews_df["label"]=='__label__xuat_sac'])
tot = len(reviews_df[reviews_df["label"]=='__label__tot'])
trung_binh = len(reviews_df[reviews_df["label"]=='__label__trung_binh'])
kem = len(reviews_df[reviews_df["label"]=='__label__kem'])
rat_kem =len( len(reviews_df[reviews_df["label"]=='__label__rat_kem']))
#// ve bieu do
divisions = ['xuat_xac','tot','trung_binh','kem',"rat_kem"]
divisions_marks = [xuat_xac,tot,trung_binh,kem]
plt.bar(divisions,divisions_marks,color="green")
plt.title("thong ke so luong")
plt.xlabel('label')
plt.ylabel('so luong ban gi')
plt.show()



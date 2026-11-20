"""Proxy metrics for the "similar books" (KNN) recommendations.

For 500 random books, compares their 10 neighbours (other editions of the
book left out, as in the app) with 10 random books on:
- same author: share of recommendations written by the same author
- tags Jaccard: overlap of the final tags
- shared readers: share of recommendations liked (rating >= 8) by at least one
  reader who also liked the query book. This signal is not used by the model.

Usage: python evaluate_knn.py [artifacts_dir]
"""
import pickle
import sys

import numpy as np
import pandas as pd

from recommender import EditionIndex, similar_books

art = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
knn = pickle.load(open(f"{art}/knn_model.pkl","rb"))
X = pickle.load(open(f"{art}/X_final.pkl","rb"))
bk = pickle.load(open(f"{art}/book_df_knn.pkl","rb"))
d = pd.read_csv("data/dataset_final3.csv", low_memory=False)
readers = d[d["Book-Rating"]>=8].groupby("ISBN")["User-ID"].apply(set).to_dict()
tags = {r.ISBN: set(t.strip() for t in str(r.Final_Tags).split(",")) for r in bk.itertuples()}
titles = bk["Book-Title"].tolist()
isbn_of = dict(zip(titles, bk.ISBN))
editions = EditionIndex(titles, bk["Book-Author"])
auth = dict(zip(bk.ISBN, bk["Book-Author"].str.lower().str.strip()))
rng = np.random.default_rng(0)
idx = rng.choice(len(bk), 500, replace=False)
K=10
def stats(pairs):
    sa=jac=co=0; n=0
    for q,r in pairs:
        n+=1
        sa += auth[q]==auth[r]
        a,b=tags[q],tags[r]; jac += len(a&b)/len(a|b)
        co += len(readers.get(q,set()) & readers.get(r,set()))>0
    return sa/n, jac/n, co/n
pairs=[]; rand=[]
for i in idx:
    q = bk.ISBN.iloc[i]
    # Same selection as the app: other editions of the book are left out.
    recs = [isbn_of[t] for t in similar_books(i, knn, X, titles, editions, n=K)]
    pairs += [(q,r) for r in recs]
    rand += [(q,bk.ISBN.iloc[j]) for j in rng.choice(len(bk),K)]
m, b = stats(pairs), stats(rand)
print(f"{'':22s}{'model':>9s}{'random':>9s}")
for name,x,y in zip(["same author","tags Jaccard","shared readers"],m,b):
    print(f"{name:22s}{x:9.3f}{y:9.3f}")

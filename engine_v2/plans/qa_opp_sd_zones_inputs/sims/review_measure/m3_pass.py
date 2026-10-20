import pickle, os
R=os.path.dirname(os.path.abspath(__file__))
h1,m15=pickle.load(open(os.path.join(R,'frames.pkl'),'rb'))
c=h1['c'].astype(float); hi=h1['h'].astype(float); lo=h1['l'].astype(float)
def first(cond, a, b):
    r=[i for i in range(a,b+1) if cond(i)]; return (r[0], len(r)) if r else None
for name,outer in [('s3c2 POI outer',0.58118),('s3c2 BOS outer',0.58196),('s3c1 POI outer',0.58318),('s3c1 BOS outer',0.58424),('s3c3 BOS outer',0.58105)]:
    print('(0,1)',name,outer,'closes > outer in 683..819:',first(lambda i: c[i]>outer,683,819), ' in 683..901:', first(lambda i: c[i]>outer,683,901))
print('(0,1) parent POI outer 0.57676 first close < :', first(lambda i: c[i]<0.57676,683,901), 'close754', c[754])
print('(0,1) closes 683..700', list(c[683:701]))
for name,outer in [('s5 POI outer',0.57276),('s5 BOS outer',0.57112),('s7c2 BOS outer',0.57691),('new(1,4183) BOS outer',0.57811),('s7c0 POI',0.57454),('s7c1 POI',0.57422)]:
    print('(1,2)',name,outer,'closes < outer in 935..1044:',first(lambda i: c[i]<outer,935,1044), 'in 935..1047', first(lambda i: c[i]<outer,935,1047))
print('(1,2) parent POI outer 0.5785 first close > :', first(lambda i: c[i]>0.5785,935,1057), 'close1047', c[1047])
print('(1,2) parent BOS outer 0.58105 first close > :', first(lambda i: c[i]>0.58105,935,1057))
for h in list(range(1040,1058)): print(h, h1.loc[h,'o'],hi[h],lo[h],c[h])

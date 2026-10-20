import pickle, os
R=os.path.dirname(os.path.abspath(__file__))
h1,m15=pickle.load(open(os.path.join(R,'frames.pkl'),'rb'))
c=h1['c'].astype(float)
print('(0,1) closes > 0.58118 idx', [i for i in range(683,902) if c[i]>0.58118])
print('(0,1) closes > 0.58196 idx', [i for i in range(683,902) if c[i]>0.58196])
print('(1,2) closes < 0.57691 idx', [i for i in range(935,1058) if c[i]<0.57691])
print('(1,2) closes < 0.57811 last', [i for i in range(935,1058) if c[i]<0.57811][-5:])

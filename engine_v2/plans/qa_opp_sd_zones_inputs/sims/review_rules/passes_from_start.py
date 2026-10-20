import pandas as pd
D="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
df=pd.read_csv(D+"NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
av=lambda m: m//4+1
# (0,1) bullish parent: opp zones above (sell), pass = close > outer
z01=[("s3c1.BOS",0.58366,0.58424,av(2829)),("s3c1.POI",0.58279,0.58318,max(av(2844),av(2829))),
     ("s3c2.BOS",0.58171,0.58196,av(2992)),("s3c2.POI",0.58096,0.58118,av(3086)),("s3c3.BOS",0.58055,0.58105,av(3589))]
for n,inner,outer,a in z01:
    pre=df.iloc[683:a]  # closes from the scan start up to (excl) the usable candle
    hits=pre.index[pre.c>outer].tolist()
    print("(0,1)",n,"usable",a,"closes>outer before usable:",len(hits), hits[:3], hits[-3:] if hits else "")
# (1,2) bearish parent: opp zones below (buy), pass = close < outer
z12=[("s5.BOS",0.57271,0.57112,av(3743)),("s5.POI",0.57306,0.57276,av(3786)),
     ("s7c0.BOS",0.57384,0.57362,av(4143)),("s7c0.POI",0.57484,0.57454,av(4143)),
     ("s7c1.BOS",0.57455,0.57413,av(4157)),("s7c1.POI",0.57462,0.57422,av(4159)),("s7c2.BOS",0.57728,0.57691,av(4179))]
for n,inner,outer,a in z12:
    pre=df.iloc[935:a]
    hits=pre.index[pre.c<outer].tolist()
    print("(1,2)",n,"usable",a,"closes<outer before usable:",len(hits), hits[:3], hits[-3:] if hits else "")
print("check iloc==index:", df.index[683], df.index[1045])
print(df.columns.tolist()[:12])

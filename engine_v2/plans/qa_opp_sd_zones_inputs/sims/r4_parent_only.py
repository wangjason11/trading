import importlib.util, sys
spec=importlib.util.spec_from_file_location("sim", sys.argv[1]); sys.argv=[sys.argv[0]]
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    sim=importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
INF=10**9
P=lambda n,i,o,f,lo: (n,i,o,lo,INF,f,None)
cycles={
 "(0,0)":(1,439,651,[P("P.BOS",0.5591,0.55808,True,439)],[P("P.CTS",0.58132,0.58246,True,439)]),
 "(0,1)":(1,683,901,[P("P.POI",0.57712,0.57676,False,683),P("P.BOS",0.57494,0.5736,True,683)],[P("P.CTS",0.58404,0.58534,True,683)]),
 "(1,0)":(-1,727,747,[P("P.BOS",0.58348,0.58424,True,727)],[P("P.CTS",0.58044,0.57902,True,727)]),
 "(1,1)":(-1,810,901,[P("P.BOS",0.58132,0.58196,True,810)],[P("P.CTS",0.57538,0.574,True,810)]),
 "(1,2)":(-1,935,1057,[P("P.POI",0.57786,0.5785,False,935),P("P.BOS",0.57996,0.58105,True,935)],[P("P.CTS",0.57282,0.57112,True,935)]),
}
for k,(d,lo,hi,s,o) in cycles.items():
    print(k, sim.run(k,d,lo,hi,s,o,False,8,show=False))

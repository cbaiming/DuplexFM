"""Derive only control flow from R51; no model or numeric evaluation."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;prior=R.parent/'round51';lock=json.loads((prior/'lock.json').read_text(encoding='utf-8'))


def source(name):
    data=(prior/name).read_bytes();assert hashlib.sha256(data).hexdigest()==lock['files'][name]
    return data.decode('utf-8')


t=source('run.py')
t=t.replace('import stability as st','import stability as st\nfrom reuse import load_fixed_sampling,stitch')
t=t.replace("'test_stability.py','pytest.xml')","'test_reuse.py','pytest.xml','reuse.py','build.py','TEST_ENVIRONMENT_REPAIR.md','pytest_attempt1_failed.xml')")
t=t.replace("suites)==10","suites)==4").replace('for number in (26,50):','for number in (50,55):')
t=t.replace("pin(B/'round50/final_readback.json');assert read(B/'round50/final_readback.json')['status']=='passed'",
    "pin(B/'round55/final_readback.json');assert read(B/'round55/final_readback.json')['passed']")
t=t.replace("read(B/'round50/lock.json')['dependencies']","read(B/'round55/lock.json')['dependencies']")
t=t.replace("pin(B/'round50/plan.json')","pin(B/'round55/plan.json')")
needle="    write(R/'lock.json',dict(files="
extra='''    prior=B/'round51';receipt=read(prior/'closure_receipt.json')
    assert receipt['status']=='closed' and receipt['readback_passed']
    for name,h in receipt['files_sha256'].items():assert sha(B/name)==h,name
    assert read(prior/'verification.json')['independent_original_graph']
    for name in ('lock.json','closure_receipt.json','verification.json','final_readback.json','stability.py','results/complete.json'):
        pin(prior/name)
    for name in ('groups_fixed.json','statistics.npz'):
        pin(prior/'results'/name,read(prior/'results/complete.json')['files_sha256'][name])
'''
assert needle in t;t=t.replace(needle,extra+needle)
t=t.replace("B/'round50/results/oof_predictions.csv',B/'round26/results/oof_predictions.csv'",
    "B/'round55/results/oof_predictions.csv',B/'round50/results/oof_predictions.csv'")
needle="    rows=input_rows();labels=st.components([r['guide'] for r in rows],[r['target57'] for r in rows]);K=int(labels.max()+1)"
assert needle in t;t=t.replace(needle,"    rows=input_rows();prior=B/'round51/results'\n    labels,counts,refs=load_fixed_sampling(prior,[r['sample_id'] for r in rows],read(prior/'complete.json')['files_sha256']);K=int(labels.max()+1)")
t=t.replace("input_only=True,before_numeric_labels=True,lock_sha256=sha(R/'lock.json')",
    "input_only=True,before_numeric_labels=True,reused_groups_sha256=sha(prior/'groups_fixed.json'),reused_statistics_sha256=sha(prior/'statistics.npz'),lock_sha256=sha(R/'lock.json')")
t=t.replace("        pc[j],ms[j]=st.statistics(y,p);observed[j]=st.paired(pc[j],ms[j])",
    "        npcc,nmse=st.statistics(y,p[:,1:5]);pc[j]=stitch(npcc,refs['observed_PCC'][j]);ms[j]=stitch(nmse,refs['observed_MSE'][j]);observed[j]=st.paired(pc[j],ms[j])")
needle="            w=(labels[ix]!=g).astype(float);removed[g,j]=int(len(w)-w.sum());loopc[g,j],looms[g,j]=st.statistics(y,p,w);loo[g,j]=st.paired(loopc[g,j],looms[g,j])"
assert needle in t;t=t.replace(needle,"            w=(labels[ix]!=g).astype(float);removed[g,j]=int(len(w)-w.sum());npcc,nmse=st.statistics(y,p[:,1:5],w)\n            loopc[g,j]=stitch(npcc,refs['leave_PCC'][g,j]);looms[g,j]=stitch(nmse,refs['leave_MSE'][g,j]);loo[g,j]=st.paired(loopc[g,j],looms[g,j])")
t=t.replace("    counts=np.random.default_rng(SEED).multinomial(K,np.full(K,1/K),size=DRAWS)\n",'')
t=t.replace("                bootpc[draw,j],bootms[draw,j]=st.statistics(y,p,weights[labels[ix]])",
    "                npcc,nmse=st.statistics(y,p[:,1:5],weights[labels[ix]])\n                bootpc[draw,j]=stitch(npcc,refs['bootstrap_PCC'][draw,j]);bootms[draw,j]=stitch(nmse,refs['bootstrap_MSE'][draw,j])")
t=t.replace('paired_Mix_weights_shared=True,no_new_model=True',
    'paired_Mix_weights_shared=True,reused_R51_groups=True,reused_R51_draws=True,reused_reference_model_statistics=2,new_condition_statistics=4,no_new_model=True')
for bad in ('st.components','random.default_rng','round26/results','test_stability.py'):assert bad not in t,bad
compile(t,str(R/'run.py'),'exec')
with (R/'run.py').open('x',encoding='utf-8') as f:f.write(t)

t=source('verify.py')
start=t.index('    spec=importlib.util.spec_from_file_location');end=t.index("    with np.load(root/'statistics.npz'",start)
t=t[:start]+'''    prior=io.B/'round51';pv=io.read(prior/'verification.json');pg=io.read(prior/'results/groups_fixed.json')
    assert pv['passed'] and pv['independent_original_graph'] and pg['ids']==[r['sample_id'] for r in rows]
    g=np.array(pg['labels'],int);K=pg['global_components'];assert K==pv['global_components']
    saved=io.read(root/'groups_fixed.json');assert saved['ids']==pg['ids'] and saved['global_components']==K and saved['input_only'] and saved['before_numeric_labels']
    assert saved['reused_groups_sha256']==io.sha(prior/'results/groups_fixed.json') and saved['reused_statistics_sha256']==io.sha(prior/'results/statistics.npz')
    close(saved['labels'],g)
''' +t[end:]
t=t.replace("io.B/'round50/results/oof_predictions.csv'),io.table(io.B/'round26/results/oof_predictions.csv')",
    "io.B/'round55/results/oof_predictions.csv'),io.table(io.B/'round50/results/oof_predictions.csv')")
t=t.replace("previous=io.read(io.B/'round50/verification.json')['goal_summary']","previous=io.read(io.B/'round55/verification.json')['goal_summary']")
t=t.replace("assert a['macro'][0]<5","assert a['macro'][0]>=5")
t=t.replace("independent_original_graph=True,global_components=K,direct_edges=int(adj.sum()//2)",
    "reused_independently_verified_graph=True,new_graph_computations=0,global_components=K,direct_edges=pv['direct_edges']")
needle="    close(a['labels'],g);d=io.read(root/'diagnostic.json');assert d['global_components']==K"
extra='''
    with np.load(prior/'results/statistics.npz',allow_pickle=False) as old:
        for key in ('observed_PCC','observed_MSE','leave_PCC','leave_MSE','bootstrap_PCC','bootstrap_MSE'):
            close(a[key][...,[0,5]],old[key][...,[0,1]])
        close(a['counts'],old['counts'])
    assert d['reused_R51_groups'] and d['reused_R51_draws'] and d['reused_reference_model_statistics']==2 and d['new_condition_statistics']==4
'''
assert needle in t;t=t.replace(needle,needle+extra)
for bad in ('native.build_adjacency','round26/results',"int(adj.sum()//2)","a['macro'][0]<5"):assert bad not in t,bad
compile(t,str(R/'verify.py'),'exec')
with (R/'verify.py').open('x',encoding='utf-8') as f:f.write(t)
print('R56 derived from locked R51, changing fixed result identities and explicitly reusing sampling/reference statistics.')

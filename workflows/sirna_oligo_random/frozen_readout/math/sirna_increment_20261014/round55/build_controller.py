"""One-time audited derivation from immutable R50 control flow; no data fitting."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;origin=R.parent/'round50/run.py'
data=origin.read_bytes();expected=json.loads((origin.parent/'lock.json').read_text(encoding='utf-8'))['files']['run.py']
assert hashlib.sha256(data).hexdigest()==expected
text=data.decode('utf-8')
text=text.replace('R49','R54').replace('round49','round54').replace('round26','round50').replace('_r50_prior4','_r55_prior4').replace('_partial_group','_common_intercept')
text=text.replace("'pytest.xml')","'pytest.xml','close.py','build_controller.py')",1)
text=text.replace('gamma=.5,lambda_ratio=.1',"common_group_penalty='n/K',lambda_ratio=.1")
text=text.replace('total_training_partitions=96,reused_R54_partitions=15,new_training_partitions=81',
    'total_training_partitions=96,reused_R50_partitions=96,new_training_partitions=0,reused_R54_geometries=15,new_group_geometries=81')
start=text.index('def group_block(');end=text.index('\ndef run(',start)
text=text[:start]+'''def group_block(item,rows,fi,root):
    result=[];s,f=item['setting'],item['fold'];out=root/'groups'/s/f'fold{f}'
    for j in range(6):
        tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);targets=[rows[fi[i]]['target57'] for i in tr]
        name='final' if j==5 else f'inner{j}';origin=N/'results/groups'/s/f'fold{f}'/(name+'.json');saved=io.read(origin)
        assert saved['train_indices']==tr.tolist() and saved['train_ids']==[item['fit_ids'][i] for i in tr]
        assert saved['target_strings_sha256']==hashlib.sha256('\\n'.join(targets).encode()).hexdigest()
        g=np.array(saved['labels'],int);pilot=None
        if f==0 and s in SETTINGS[:3] and j<5:
            pilot=Q/'pilot/geometry'/s/name/'geometry.json';oldmeta=io.read(pilot)
            assert oldmeta['train_indices']==tr.tolist() and oldmeta['train_ids']==[item['fit_ids'][i] for i in tr]
            meta=oldmeta['metadata']
        else:meta=head.geometry(g)
        state=dict(train_indices=tr,train_ids=[item['fit_ids'][i] for i in tr],labels=g,metadata=meta,
            source_group_metadata=saved['metadata'],target_strings_sha256=saved['target_strings_sha256'],
            reused_R50_sha256=io.sha(origin),reused_R54_geometry_sha256=io.sha(pilot) if pilot else None)
        io.write(out/(name+'.json'),state);result.append((g,meta,out/(name+'.json')))
    io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
    return result

''' +text[end:]
text=text.replace("('inner_new','inner_R54','inner_R26_identity','inner_TD_identity','final_new','final_R26_identity','final_TD_identity')",
    "('inner_new','inner_R54','inner_R50_equal','final_new','final_R50_equal')")
start=text.index("        basepath=old.OLD/'results/folds'");end=text.index('        for source in SOURCES:',start);text=text[:start]+text[end:]
text=text.replace("nt=N/'results/folds'/s/f'fold{f}'/origin","nt=N/'results/folds'/s/f'fold{f}'/source")
line="            with np.load(ref/'arrays.npz',allow_pickle=False) as a:tdcoef=a['inner_coef'][:,:5,:24].copy();tdcenter=a['inner_intercept'].copy()\n"
assert line in text;text=text.replace(line,'',1)
start=text.index('                    kind=head.reuse_kind(');end=text.index("                    counts['inner_'+kind]",start)
text=text[:start]+'''                    kind=head.reuse_kind(s,f,j,alpha,meta['reference_equal']);origin_files={}
                    if kind=='R54':
                        folder=Q/'pilot'/s/source/f'inner{j}';st=io.read(folder/'state.json');cert=st['certificate']
                        assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                        assert st['geometry_sha256']==io.sha(Q/'pilot/geometry'/s/f'inner{j}'/'geometry.json')
                        with np.load(folder/'model.npz',allow_pickle=False) as a:b=a['beta'].copy();c=float(a['intercept'])
                        origin_files={str((folder/n).resolve()):io.sha(folder/n) for n in ('model.npz','state.json','complete.json')}
                    elif kind=='R50_equal':
                        with np.load(nt/'arrays.npz',allow_pickle=False) as a:b=a['inner_coef'][j,k].copy();c=float(a['inner_intercept'][j,k])
                        cert=io.read(nt/f'inner{j}_alpha{k}_certificate.json')
                        origin_files={str((nt/n).resolve()):io.sha(nt/n) for n in ('arrays.npz',f'inner{j}_alpha{k}_certificate.json')}
                    else:b,c,cert=head.fit(z,yf[tr],labels,alpha,is_source,deadline=start+1800)
''' +text[end:]
start=text.index("            oldalpha=io.read(nt/'selection.json')");end=text.index("            counts['final_'+kind]",start)
text=text[:start]+'''            oldalpha=io.read(nt/'selection.json')['alpha']
            kind=head.reuse_kind(s,f,None,alpha,meta['reference_equal'],oldalpha);origin_files={}
            if kind=='R50_equal':
                with np.load(nt/'arrays.npz',allow_pickle=False) as a:b=a['final_coef'].copy();c=float(a['final_intercept'])
                cert=io.read(nt/'final_certificate.json')
                origin_files={str((nt/n).resolve()):io.sha(nt/n) for n in ('arrays.npz','final_certificate.json','selection.json')}
            else:b,c,cert=head.fit(z,yf,labels,alpha,is_source,deadline=start+1800)
''' +text[end:]
text=text.replace('final_identity_reused','final_equal_reused').replace('paired_R26.csv','paired_R50.csv')
text=text.replace('identity_reuse=True','equal_group_reuse=True')
text=text.replace("N/'verify.py',old.OLD/'results/summary.csv'","N/'verify.py',B/'round26/head.py',B/'round26/verify.py',old.OLD/'results/summary.csv'")
text=text.replace("'close.py','build_controller.py')","'close.py','build_controller.py','audit_math.py','build_verifier.py')")
for bad in ('R26_identity','TD_identity','basealpha','head.td_cert','identity_on_centered_space','reused_R54_partitions','new_training_partitions=81'):
    assert bad not in text,bad
compile(text,str(R/'run.py'),'exec')
with (R/'run.py').open('x',encoding='utf-8') as f:f.write(text)
print('Derived controller from',expected)

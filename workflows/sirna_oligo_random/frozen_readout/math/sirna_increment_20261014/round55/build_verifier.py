"""Source-only derivation of the complete R50 audit, before registration."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;origin=R.parent/'round50/verify.py'
data=origin.read_bytes();expected=json.loads((origin.parent/'lock.json').read_text(encoding='utf-8'))['files']['verify.py']
assert hashlib.sha256(data).hexdigest()==expected
text=data.decode('utf-8').replace("Q=B/'round49';N=B/'round26'","Q=B/'round54';N=B/'round50'")
text=text.replace("prev=load('_r50_audit26',N/'verify.py');group=load('_r50_audit49',Q/'verify.py')",
    "prev=load('_r55_audit26',B/'round26/verify.py');group=load('_r55_audit49',B/'round49/verify.py');math=load('_r55_math',R/'audit_math.py')")
start=text.index('def audit_TD(');end=text.index('def main():',start);text=text[:start]+text[end:]
text=text.replace("plan['gamma']==.5","plan['common_group_penalty']=='n/K'")
text=text.replace('_partial_group','_common_intercept').replace('paired_R26.csv','paired_R50.csv')
text=text.replace("('inner_new','inner_R49','inner_R26_identity','inner_TD_identity','final_new','final_R26_identity','final_TD_identity')",
    "('inner_new','inner_R54','inner_R50_equal','final_new','final_R50_equal')")
start=text.index('            targets=[rows[fi[i]]');end=text.index('        for source in SOURCES:',start)
text=text[:start]+'''            targets=[rows[fi[i]]['target57'] for i in tr];labels,gm=group.independent_groups(targets)
            meta=math.indicator_geometry(labels)[3]
            assert state['train_indices']==tr.tolist() and state['train_ids']==[item['fit_ids'][i] for i in tr]
            assert state['target_strings_sha256']==hashlib.sha256('\\n'.join(targets).encode()).hexdigest();close(state['labels'],labels,0)
            group.compare_mapping(state['metadata'],meta);group.compare_mapping(state['source_group_metadata'],gm)
            prior=N/'results/groups'/s/f'fold{f}'/(name+'.json');saved=read(prior)
            assert state['reused_R50_sha256']==sha(prior) and saved['train_indices']==tr.tolist() and saved['train_ids']==state['train_ids']
            close(saved['labels'],labels,0);group.compare_mapping(saved['metadata'],gm)
            assert saved['target_strings_sha256']==state['target_strings_sha256']
            if f==0 and s in old.S[:3] and j<5:
                pilot=Q/'pilot/geometry'/s/name/'geometry.json';p=read(pilot)
                assert state['reused_R54_geometry_sha256']==sha(pilot) and p['train_indices']==tr.tolist() and p['train_ids']==state['train_ids']
                group.compare_mapping(p['metadata'],meta)
            else:assert state['reused_R54_geometry_sha256'] is None
            grouping.append((labels,meta));partition_meta.append(dict(setting=s,fold=f,stage=name,**meta));ngroup+=1
''' + text[end:]
text=text.replace("nt=N/'results/folds'/s/f'fold{f}'/origin","nt=N/'results/folds'/s/f'fold{f}'/source")
start=text.index("            with np.load(ref/'arrays.npz'");end=text.index('            oof=np.full',start);text=text[:start]+text[end:]
start=text.index('                zz=group.weighted_design');end=text.index('                for k,alpha',start)
text=text[:start]+"                profiled=math.profile(z if is_source else z[:,:24],y[fi[tr]],labels)\n"+text[end:]
text=text.replace("if f==0 and s in old.S[:3] and alpha==100.:kind='R49'\n                    elif meta['identity_on_centered_space']:kind='R26_identity' if is_source else 'TD_identity'",
    "if f==0 and s in old.S[:3] and alpha==100.:kind='R54'\n                    elif meta['reference_equal']:kind='R50_equal'")
text=text.replace("if kind=='R49':","if kind=='R54':").replace("elif kind=='R26_identity':","elif kind=='R50_equal':")
text=text.replace("                            assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()",
    "                            assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()\n                            assert st['geometry_sha256']==sha(Q/'pilot/geometry'/s/f'inner{j}'/'geometry.json')")
text=text.replace("                        else:close(b,tdcoef[j,k],0);close(c,tdcenter[j],0)\n",'')
text=text.replace("normal,gap=prev.check_head(zz,yy,b,c,cert,alpha,gram,rhs,reused='maximum_iterations' not in cert)",
    "normal,gap=math.check_source(z,y[fi[tr]],labels,b,c,cert,alpha,profiled)")
text=text.replace("normal,error=audit_TD(zz,yy,b,c,cert,alpha)",
    "normal,error=math.audit_TD(z[:,:24],y[fi[tr]],labels,b,c,cert,alpha,profiled)",1)
start=text.index('            zz=group.weighted_design');end=text.index('            assert len(reuse)==expected_reuse',start)
text=text[:start]+'''            profiled=math.profile(z if is_source else z[:,:24],y[fi],labels)
            b,c=a['final_coef'],float(a['final_intercept']);cert=read(out/'final_certificate.json');oldalpha=read(nt/'selection.json')['alpha']
            kind='R50_equal' if meta['reference_equal'] and alpha==oldalpha else 'new'
            assert references['final_equal_reused']==(kind!='new');counts['final_'+kind]+=1
            if kind!='new':
                rr=next(r for r in reuse if r['stage']=='final');assert rr['kind']==kind and rr['candidate']==k;expected_reuse+=1
                with np.load(nt/'arrays.npz',allow_pickle=False) as cm:close(b,cm['final_coef'],0);close(c,cm['final_intercept'],0)
                assert cert==read(nt/'final_certificate.json')
''' +text[end:]
text=text.replace("normal,gap=prev.check_head(zz,yy,b,c,cert,alpha,zz.T@zz,zz.T@(yy-yy.mean()),reused='maximum_iterations' not in cert)",
    "normal,gap=math.check_source(z,y[fi],labels,b,c,cert,alpha,profiled)")
text=text.replace("normal,error=audit_TD(zz,yy,b,c,cert,alpha)","normal,error=math.audit_TD(z[:,:24],y[fi],labels,b,c,cert,alpha,profiled)")
text=text.replace("counts['inner_R49']","counts['inner_R54']")
text=text.replace("complete['new_training_partitions']==81 and complete['reused_R49_partitions']==15",
    "complete['new_training_partitions']==0 and complete['reused_R50_partitions']==96 and complete['reused_R54_geometries']==15 and complete['new_group_geometries']==81")
for bad in ('R26_identity','TD_identity','identity_on_centered_space','group.weighted_design','prev.check_head','reused_R49','inner_R49','tdcoef','basealpha'):
    assert bad not in text,bad
compile(text,str(R/'verify.py'),'exec')
with (R/'verify.py').open('x',encoding='utf-8') as f:f.write(text)
print('Derived verifier from',expected)

"""Copy audited orchestration, explicitly substitute the new mathematical contract."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'round29'
lock=json.loads((P/'lock.json').read_text(encoding='utf-8'))


def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)


for name in ('run.py','verify.py'):
    assert hashlib.sha256((P/name).read_bytes()).hexdigest()==lock['files'][name]
    text=(P/name).read_text(encoding='utf-8')
    if name=='run.py':
        text=replace(text,'Bounded TD-coefficient trust experiment on unchanged native source inputs.','Bounded grouped-channel readout on unchanged native source inputs.')
        text=replace(text,'from anchored import','from channel_sparse import')
        text=replace(text,"'_r29_prior4'","'_r30_prior4'")
        text=replace(text,"'anchored.py'","'channel_sparse.py'")
        text=replace(text,"'test_anchored.py'","'test_channel_sparse.py'")
        text=replace(text,"'CODE_AUDIT.md','pytest.xml')","'CODE_AUDIT.md','pytest.xml','prepare_r30.py')")
        text=replace(text,"round28/closure_receipt.json","round29/closure_receipt.json",2)
        text=replace(text,"alpha=100.,kappa=100.,lambda_ratio=.1,matrix_shape=[6,32]",
            "alpha=100.,lambda_ratio=.1,matrix_shape=[6,32],group_axis='columns',group_count=32,group_size=6")
        text=replace(text,"fit(z[tr],y[tr],b0[j],deadline=start+900)","fit(z[tr],y[tr],deadline=start+900)")
        text=replace(text,"rank=cert['rank'],iterations=cert['iterations']",
            "active_channels=cert['active_channels'],effective_channels=cert['effective_channels'],iterations=cert['iterations']")
    else:
        text=replace(text,'Independent augmented-system certificate for a fixed TD coefficient anchor.',
            'Independent full-system certificate for fixed native channel group sparsity.')
        text=replace(text,"'_r29_independent4'","'_r30_independent4'")
        text=replace(text,"lock['alpha']==lock['kappa']==100","lock['alpha']==100")
        text=replace(text,"    assert lock['settings']==list(old.S[:3])", "    assert lock['group_axis']=='columns' and lock['group_count']==32 and lock['group_size']==6\n    assert lock['settings']==list(old.S[:3])")
        text=replace(text,"                close(cert['TD_reference_error'],referr,1e-10)\n",'')
        text=replace(text,"                g[:24,:24]+=100*np.eye(24);rhs[:24]+=100*b0\n",'')
        text=replace(text,"critical=float(np.linalg.norm(q.reshape(6,32),2))", "critical=float(max(np.linalg.norm(q[channel::32]) for channel in range(32)))")
        text=replace(text,"                close(ns['certificate']['lambda_critical'],critical,1e-10)\n",'')
        text=replace(text,"cert['alpha']==cert['kappa']==100","cert['alpha']==100")
        text=replace(text,"cert['mode']=='TD_anchored_native_trace'", "cert['mode']=='native_channel_group'\n                assert cert['group_axis']=='columns' and cert['group_count']==32 and cert['group_size']==6")
        text=replace(text,
            "                singular=np.linalg.svd(beta[24:].reshape(6,32),compute_uv=False)\n                rank=int(sum(singular>1e-8));close(cert['singular_values'],singular,1e-11);assert cert['rank']==rank\n                dual=q-h@beta[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))\n                if norm>lam:dual*=lam/norm\n                close(cert['dual'],dual,1e-10);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-10\n                primal=.5*beta@g@beta-rhs@beta+lam*singular.sum();dv=np.r_[np.zeros(24),dual]",
            "                norms=np.array([np.linalg.norm(beta[24+channel::32]) for channel in range(32)])\n                active=int(sum(norms>0));effective=int(sum(norms>1e-8))\n                close(cert['group_norms'],norms,1e-11)\n                assert cert['active_channels']==active and cert['effective_channels']==effective\n                dual=q-h@beta[24:]\n                for channel in range(32):\n                    index=np.arange(channel,192,32);norm=float(np.linalg.norm(dual[index]))\n                    if norm>lam:dual[index]*=lam/norm\n                close(cert['dual'],dual,1e-10)\n                assert max(np.linalg.norm(dual[channel::32]) for channel in range(32))<=lam+1e-10\n                primal=.5*beta@g@beta-rhs@beta+lam*norms.sum();dv=np.r_[np.zeros(24),dual]")
        text=replace(text,"                close(cert['anchor_penalty'],50*np.sum((beta[:24]-b0)**2),1e-12)\n",'')
        text=replace(text,".5*len(tr)*train_mse+50*(beta@beta)+50*np.sum((beta[:24]-b0)**2)+lam*singular.sum()",
            ".5*len(tr)*train_mse+50*(beta@beta)+lam*norms.sum()")
        text=replace(text,"primal+.5*np.sum((y[tr]-center)**2)+50*(b0@b0)","primal+.5*np.sum((y[tr]-center)**2)")
        text=replace(text,"rank=rank,iterations=cert['iterations']", "active_channels=active,effective_channels=effective,iterations=cert['iterations']")
    ast.parse(text,filename=name)
    with (R/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    print('PREPARED',name,hashlib.sha256((R/name).read_bytes()).hexdigest())

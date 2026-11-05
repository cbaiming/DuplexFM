"""Small CPU heads, explicit endpoints, exact independent final replay."""
import hashlib,io,time
import numpy as np
import torch
from native import model,normalize_fit,normalize,loss,predict,parameters
ENDPOINTS=(8,16,32,64,100)

def state_hash(state):
    h=hashlib.sha256()
    for key,t in sorted(state.items()):
        a=t.detach().cpu().numpy();h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def clone(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def fit(x,y,condition,epochs=100,probe=None,checkpoint=None):
    # Probe is used only for prediction; no validation labels enter this function.
    started=time.monotonic();normalizer=normalize_fit(x)
    z=torch.from_numpy(normalize(x,normalizer));target=torch.as_tensor(y,dtype=torch.float32)
    m=model(condition,42);initial=clone(m);initial_hash=state_hash(initial)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator(device='cpu').manual_seed(42)
    histories=[];states={};predictions=[];updates=0
    def save(name,epoch,order,cursor):
        if checkpoint is not None:
            checkpoint(name,dict(model=clone(m),optimizer=optimizer.state_dict(),normalizer=normalizer,
                torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),epoch=epoch,
                order=order,cursor=cursor,updates=updates,condition=condition,initial_sha256=initial_hash))
    save('initial.pt',0,torch.empty(0,dtype=torch.int64),0)
    if probe is not None:xp=normalize(probe,normalizer)
    for ep in range(1,epochs+1):
        m.train();order=torch.randperm(len(z),generator=generator);total=0.
        for start in range(0,len(z),128):
            ii=order[start:start+128];optimizer.zero_grad(set_to_none=True)
            value=loss(condition,m(z[ii]),target[ii]);assert torch.isfinite(value)
            value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
            optimizer.step();total+=value.item()*len(ii);updates+=1
            if updates%500==0:save(f'update{updates}.pt',ep,order,start+len(ii))
        histories.append(dict(epoch=ep,loss=total/len(z)))
        if ep in ENDPOINTS:
            states[ep]=clone(m)
            if probe is not None:predictions.append(predict(m,xp))
    assert all(p.device.type=='cpu' for p in m.parameters())
    save('last.pt',epochs,order,len(z))
    return dict(normalizer=normalizer,initial=initial,state=clone(m),states=states,
        predictions=np.array(predictions).T if probe is not None else None,
        history=histories,initial_sha256=initial_hash,final_sha256=state_hash(m.state_dict()),
        epochs=epochs,updates=updates,trainable_parameters=parameters(m),wall_seconds=time.monotonic()-started)

def bytes_torch(value):
    buffer=io.BytesIO();torch.save(value,buffer);return buffer.getvalue()

def select(y,columns):
    yc=np.asarray(y)-np.mean(y);p=np.asarray(columns);pc=p-p.mean(0)
    den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
    r=np.divide(yc@pc,den,out=np.full(len(ENDPOINTS),-2.),where=den>1e-14)
    mse=np.mean((p-y[:,None])**2,axis=0)
    selected=min(range(len(ENDPOINTS)),key=lambda k:(-r[k],mse[k],ENDPOINTS[k]))
    return selected,r,mse

"""Bounded, read-only serialization identity audit; never imports a model."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
start=time.monotonic();prior=read(ROOT/'SOURCE43_SCOPE_PREPARATION_20261015.json')
path=Path(prior['original_config']['path']);raw=path.read_bytes();expected=prior['original_config']['expected_sha256']
assert sha_bytes(raw)==prior['original_config']['sha256']
manifest=Path(prior['manifest']['path']);mraw=manifest.read_bytes();assert sha_bytes(mraw)==prior['manifest']['sha256']
m=json.loads(mraw.decode('utf-8-sig'));assert m['source_provenance']['source_run_config_sha256']==expected
text=raw.decode('utf-8-sig');config=json.loads(text);candidates={}
def add(name,txt,bom=False):
    b=txt.encode('utf-8');b=(b'\xef\xbb\xbf'+b) if bom else b
    digest=sha_bytes(b)
    candidates.setdefault(digest,dict(names=[],bytes=len(b),matches_expected=digest==expected))['names'].append(name)
    if digest==expected:return b
    return None
matches=[]
texts={'current_decoded':text,'LF':text.replace('\r\n','\n').replace('\r','\n')}
for name,txt in list(texts.items()):
    texts[name+'_no_trailing_newline']=txt.rstrip('\r\n')
    texts[name+'_one_trailing_LF']=txt.rstrip('\r\n')+'\n'
for name,txt in list(texts.items()):
    for lineend,content in (('LF',txt.replace('\r\n','\n')),('CRLF',txt.replace('\r\n','\n').replace('\n','\r\n'))):
        for bom in (False,True):
            b=add(f'raw:{name}:{lineend}:BOM{bom}',content,bom)
            if b is not None:matches.append(b)
for ascii_only in (False,True):
    for sorted_keys in (False,True):
        for indent in (None,2,4,'\t'):
            options=[None,(',',':')] if indent is None else [None]
            for separators in options:
                rendered=json.dumps(config,ensure_ascii=ascii_only,sort_keys=sorted_keys,indent=indent,separators=separators)
                for trailing in ('','\n'):
                    for ending in ('LF','CRLF'):
                        txt=rendered+trailing
                        if ending=='CRLF':txt=txt.replace('\n','\r\n')
                        for bom in (False,True):
                            b=add(f'json:ascii{ascii_only}:sort{sorted_keys}:indent{indent!r}:sep{separators!r}:trail{bool(trailing)}:{ending}:BOM{bom}',txt,bom)
                            if b is not None:matches.append(b)
assert len(candidates)<=220
matched_bytes=matches[0] if matches else None
if matched_bytes is not None:
    assert all(x==matched_bytes for x in matches) and json.loads(matched_bytes.decode('utf-8-sig'))==config
    restored=ROOT/'source43_original_config_verified_bytes.json'
    with restored.open('xb') as f:f.write(matched_bytes)
    assert sha_bytes(restored.read_bytes())==expected
else:restored=None
out=dict(status='exact_original_bytes_recovered' if matches else 'not_resolved_by_bounded_serialization_variants',
    input_path=str(path),actual_sha256=sha_bytes(raw),expected_sha256=expected,manifest_sha256=sha_bytes(mraw),
    input_bytes=len(raw),input_has_BOM=raw.startswith(b'\xef\xbb\xbf'),input_CRLF_count=raw.count(b'\r\n'),
    top_level_keys=list(config),unique_serializations_tested=len(candidates),candidate_hashes=candidates,
    matching_hashes=[h for h,v in candidates.items() if v['matches_expected']],recovered_path=str(restored) if restored else None,
    original_file_modified=False,new_source_forward=0,model_imports=0,new_downloads=0,GPU_hours=0,
    scope_reply_received=False,export_authorized_by_this_audit=False,CPU_wall_seconds=time.monotonic()-start,
    script_sha256=sha_bytes(Path(__file__).read_bytes()))
op=ROOT/'SOURCE43_CONFIG_BYTE_AUDIT_20261015.json'
with op.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
assert path.read_bytes()==raw and manifest.read_bytes()==mraw
status='找到与原清单逐字节SHA一致的序列化，另存恢复副本。原配置未覆盖。' if matches else '测试的换行/BOM/常见JSON缩进、键排序与分隔符组合均未恢复原清单SHA。该差异仍未解释。'
md=f'''# source43配置字节审计

2026-10-15。{status}

只对已知run_config.json进行{len(candidates)}个独特序列化的SHA比较；所有变体解析后保持当前JSON值，不替换任何路径或参数。不读取测试指标，不加载模型，不下载或启动前向。原文件读前读后字节一致。

当前SHA `{sha_bytes(raw)}`；原feature manifest预期 `{expected}`。原feature manifest SHA `{sha_bytes(mraw)}`。

{'精确恢复的副本路径：'+str(restored) if restored else '没有原始配置字节或其它版本作语义对照，所以不能确定差异是路径重写、元数据变化还是模型相关字段变化。未命中有限序列化不等于排除了所有纯格式变化，更不等于证明模型架构被更改。'}

该准备没有授权source43导出，范围回复仍待用户提供。当前目标继续只用已核验seed42缓存。既有source43权重已核验的事实不变，但完整CPU推理资格仍需原生配置/输入/依赖审计，不能以文件存在替代验收。

CPU字节审计{out['CPU_wall_seconds']:.6f}秒；新GPU、来源forward、模型导入、下载均0。机器读取证据：`SOURCE43_CONFIG_BYTE_AUDIT_20261015.json`，SHA `{sha_bytes(op.read_bytes())}`。
'''
with (ROOT/'SOURCE43_CONFIG_BYTE_AUDIT_20261015.md').open('x',encoding='utf-8') as f:f.write(md)
print(json.dumps({k:out[k] for k in ('status','unique_serializations_tested','matching_hashes','original_file_modified','new_source_forward','CPU_wall_seconds')}))

"""Static archive metadata only; no pickle execution or model runtime."""
import hashlib,json,pickletools,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();prior=read(R/'SOURCE43_SCOPE_PREPARATION_20261015.json');path=Path(prior['checkpoint']['path'])
assert sha(path)==prior['checkpoint']['sha256']
with zipfile.ZipFile(path) as z:
    names=[n for n in z.namelist() if n=='data.pkl' or n.endswith('/data.pkl')];assert len(names)==1
    inf=z.getinfo(names[0]);assert inf.file_size<2*1024*1024;raw=z.read(names[0]);ops=list(pickletools.genops(raw))
strings=[arg for op,arg,pos in ops if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8')]
keys=('run_config','config','args','cli_args','metadata','source_provenance','architecture','foundation_config','epoch')
exact={k:strings.count(k) for k in keys}
json_strings=[]
for s in strings:
    if s.lstrip().startswith(('{','[')):
        try:value=json.loads(s)
        except json.JSONDecodeError:continue
        json_strings.append(dict(bytes=len(s.encode('utf-8')),kind=type(value).__name__,keys=list(value)[:20] if isinstance(value,dict) else None))
record=dict(checkpoint_path=str(path),checkpoint_sha256=sha(path),pickle_member=names[0],pickle_bytes=len(raw),
    pickle_sha256=hashlib.sha256(raw).hexdigest(),exact_metadata_key_counts=exact,string_count=len(strings),
    largest_string_bytes=max((len(s.encode('utf-8')) for s in strings),default=0),embedded_JSON_string_candidates=json_strings,
    global_symbol_names=sorted(set(arg for op,arg,pos in ops if op.name=='GLOBAL')),
    original_config_recovered=False,pickle_executed=False,torch_imported=False,model_imported=False,tensor_storage_bytes_read=0,
    new_source_forward=0,GPU_hours=0,new_downloads=0,scope_reply_received=False,CPU_wall_seconds=time.monotonic()-start,script_sha256=sha(Path(__file__)))
write(R/'SOURCE43_CHECKPOINT_METADATA_KEYS_20261015.json',record)
byte=read(R/'SOURCE43_CONFIG_BYTE_AUDIT_20261015.json')
assert sha(Path(byte['input_path']))==byte['actual_sha256'] and byte['original_file_modified'] is False
cl=read(R/'round23/closure_receipt.json')
for n,h in cl['files_sha256'].items():assert sha(R/n)==h,n
write(R/'SOURCE43_IDENTITY_FOLLOWUP_READBACK_20261015.json',dict(passed=True,new_source_forward=0,GPU_hours=0,
    files_sha256={n:sha(R/n) for n in ('SOURCE43_CONFIG_BYTE_AUDIT_20261015.json','SOURCE43_CONFIG_BYTE_AUDIT_20261015.md',
        'SOURCE43_CHECKPOINT_METADATA_KEYS_20261015.json','source43_readonly_identity_followup.py','round23/closure_receipt.json')},
    source_config_compatible_not_proven=True,scope_reply_pending=True))
note=f'''# source43身份准备的静态补充

2026-10-15。已核验的16,425,027字节checkpoint仍匹配原SHA。只读取ZIP内53,440字节data.pkl并用pickletools查看指令，未反序列化/执行pickle，未读取张量存储、导入Torch或加载模型。

标准配置键run_config/config/args/cli_args/metadata/source_provenance/architecture/foundation_config的出现次数为{[exact[k] for k in keys[:-1]]}；epoch出现{exact['epoch']}次。完整字符串中能解析的内嵌JSON候选为{len(json_strings)}。没有从这个静态键审计恢复原配置；这不是对所有可能编码格式作不存在证明。

此前72种常见JSON序列化均未恢复清单预期SHA，两个证据合起来仍不能判断具体语义差异。配置可能曾为移植而修改路径，也可能含其它变化，未获原始版本前不能指定原因。原文件和权重均未修改，尚不能宣称source43推理已就绪。

本次仅为未来桥接准备，source43新导出范围仍待用户答复；当前优化继续限定已有source42冻结缓存，没有GPU、forward或下载。CPU静态检查{record['CPU_wall_seconds']:.6f}秒。证据与独立文件读回在SOURCE43_CHECKPOINT_METADATA_KEYS_20261015.json、SOURCE43_IDENTITY_FOLLOWUP_READBACK_20261015.json。
'''
with (R/'SOURCE43_IDENTITY_FOLLOWUP_20261015.md').open('x',encoding='utf-8') as f:f.write(note)
with (R/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write('\nR23后补充只读source43身份准备：72种有限格式变体均未匹配原configSHA；ZIP的53,440B data.pkl静态指令未找到标准config/args键或内嵌JSON。未执行pickle、读张量、导入模型或新forward，不能据此证明语义兼容或全局原配置不存在。两证据文件及R23closure SHA独立读回通过，见SOURCE43_IDENTITY_FOLLOWUP_20261015.md。范围问题仍待回复，不把它视为当前长期阻断。\n')
print(json.dumps(dict(static_metadata_saved=True,exact_metadata_keys=exact,embedded_JSON_candidates=len(json_strings),new_source_forward=0,identity_readback_passed=True)))

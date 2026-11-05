"""Describe independently verified R11; no fitting or prediction recomputation."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'round11'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert not (R/'postresult_comparison.json').exists() and not (ROOT/'ROUND11_RESULTS.md').exists()
    verification=read(R/'verification.json');assert verification['passed']
    assert verification['complete_sha256']==sha(R/'results/complete.json')
    complete=read(R/'results/complete.json');summary=table(R/'results/summary.csv')
    goals=read(R/'results/goal_summary.json');selected=table(R/'results/selection.csv');plan=read(R/'plan.json')
    lookup={(x['setting'],x['condition']):x for x in summary};settings=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
    paired=[];usage=[]
    for view in ('core_real','full_real','initial_core'):
        for mode in ('AlwaysARD','AdaptiveARD'):
            for setting in settings:
                a=lookup[setting,view+'_'+mode];b=lookup[setting,'TD_only_'+mode];fresh=lookup[setting,'initial_core_'+mode]
                paired.append(dict(setting=setting,view=view,mode=mode,
                    PCC_delta_vs_matched_TD_ARD=float(a['PCC'])-float(b['PCC']),
                    PCC_relative_pct_vs_matched_TD_ARD=100*(float(a['PCC'])/float(b['PCC'])-1),
                    PCC_delta_vs_initial=float(a['PCC'])-float(fresh['PCC'])))
    for view in ('core_real','full_real','initial_core','TD_only'):
        for setting in settings:
            part=[x for x in selected if x['view']==view and x['setting']==setting and x['condition'].endswith('_AdaptiveARD')]
            used={int(x['fold']) for x in part if int(x['candidate'])==5}
            count=sum(len(f['test_ids']) for f in plan['folds'] if f['setting']==setting and f['fold'] in used)
            models=[read(R/'results/folds'/setting/f"fold{x['fold']}"/view/'models.json')['final'] for x in part]
            usage.append(dict(setting=setting,view=view,ARD_folds=len(used),total_folds=len(part),ARD_rows=count,
                median_active_TD=statistics.median(m['active_TD'] for m in models),
                median_active_source=statistics.median(m['active_source'] for m in models),
                final_iteration_range=[min(m['iterations'] for m in models),max(m['iterations'] for m in models)]))
    execution=[read(p) for p in R.glob('execution_*.json')];audit_execution=[read(p) for p in R.glob('verification_execution_*.json')]
    assert len(execution)==len(audit_execution)==1 and execution[0]['returncode']==audit_execution[0]['returncode']==0
    result=dict(paired=paired,usage=usage,all_eight_conditions_below_5pct=all(not x['reaches_development_5pct'] for x in goals),
        run_CPU_seconds=execution[0]['CPU_wall_seconds'],verification_total_CPU_seconds=audit_execution[0]['CPU_wall_seconds'],
        iteration_capped_fits=complete['iteration_capped_fits'],new_training_fits=384,verification_replay_fits=64,
        new_source_forward=0,GPU_hours=0,verification_sha256=sha(R/'verification.json'),complete_sha256=sha(R/'results/complete.json'))
    write(R/'postresult_comparison.json',result)
    lines=['# R11：自动相关性判定未改善主方案','',
        '2026-10-15，固定CPU实验与独立验收均完成。主方案Core AdaptiveARD对原TD24 Ridge的四设置相对PCC均值为 **+0.9015%**，低于5%目标与R4的+2.6812%。不采用、不扩ARD超参数或事后改主方案。','',
        '| 设置 | 原TD24 PCC | Core AdaptiveARD PCC | 相对PCC变化 | MSE相对下降 |',
        '|---|---:|---:|---:|---:|']
    for s in settings:
        r=lookup[s,'core_real_AdaptiveARD'];lines.append(f"| {s} | {float(r['TD24_PCC']):.6f} | {float(r['PCC']):.6f} | {float(r['PCC_relative_gain_pct']):+.4f}% | {float(r['MSE_relative_reduction_pct']):+.4f}% |")
    lines+=['','| 条件 | 四设置平均相对PCC变化 |','|---|---:|']
    for g in goals:lines.append(f"| {g['condition']} | {g['mean_relative_PCC_gain_pct']:+.4f}% |")
    lines+=['',
        '固定使用Core ARD也只有+2.1667%，Mixset−5.4385%、跨队列−0.5634%；因此不能只解释为gate过于保守。它在Takayuki达到+13.8476%，同时同算法TD-only也有+4.0087%，不能把全部改善归因来源。','',
        '真实miRNA训练不是本轮所有条件中最好的来源：初始化AlwaysARD平均+2.5720%，高于训练后Core的+2.1667%；初始化AdaptiveARD+1.1455%也高于Core主方案+0.9015%。这不证明miRNA监督普遍有害，但本轮不支持稳定的监督增量；精确初始化本身仍包含公共基础模型。','',
        '| Core Adaptive选择 | ARD外折/总折 | 使用ARD的测试行 | 最终ARD中TD有效系数中位数 | 来源有效系数中位数 |',
        '|---|---:|---:|---:|---:|']
    for u in usage:
        if u['view']=='core_real':lines.append(f"| {u['setting']} | {u['ARD_folds']}/{u['total_folds']} | {u['ARD_rows']} | {u['median_active_TD']} | {u['median_active_source']} |")
    lines+=['',
        '“有效”仅指原生ARD精度阈值未剪枝，不是生物重要性或可靠因果效应。零source系数时才可说该最终头不使用来源；选择TD时逐行复用原强基线，不平均或叠加模型预测。完整配对同算法TD/初始化差异在round11/postresult_comparison.json。','',
        f"新拟合320个内层ARD＋64个最终ARD，输出8条件30384行和128选择；独立验收另重复64个最终训练，总实际ARD拟合448次，不能只报64。全部后验方程、R4 PCA/scaler训练来源、SHA、逐行预测、候选选择、raw216部署和完整指标通过；最终重放系数与原生预测差0，384个科学拟合达到300步上限的数量为{complete['iteration_capped_fits']}。算法数值验收不代表全局证据最优。","", 
        f"控制器CPU本体{execution[0]['CPU_wall_seconds']:.3f}秒，独立验收总本体{audit_execution[0]['CPU_wall_seconds']:.3f}秒（含64次重放，不与其内部计时重复相加）。合成7项成功0.88秒；首轮合成错误断言失败1.02秒及旧代码保留。进程启动、开发和既有来源训练/缓存成本另计，不是端到端几秒。GPU/新来源前向均0。",'',
        '方法参考[Tipping 2001](https://www.jmlr.org/papers/v1/tipping01a.html)及[官方ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)，本轮是固定库实现的线性特征ARD，不是样本核RVM。','',
        f"科学锁SHA `{sha(R/'lock.json')}`；complete SHA `{sha(R/'results/complete.json')}`；独立验收SHA `{sha(R/'verification.json')}`。",'',
        '源keep-overlap、context暴露、单来源42、已反复开发四设置、Mix与跨队列共享测试队列等限制不变；所有结果属开发证据。当前未达5%、没有新盲确认或SOTA。下一轮不可重复ARD网格或按每个test挑最优源/策略。']
    with (ROOT/'ROUND11_RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write(ROOT/'GOAL_TURN_R10_R11_AUDIT.json',dict(goal_status='active',goal_achieved=False,previous_goal_turn='progress',
        this_goal_turn='progress',R10_qualified=False,R11_primary_mean_relative_PCC_gain_pct=.9015169479602709,
        all_eight_conditions_below_5pct=result['all_eight_conditions_below_5pct'],no_running_process=True,blocking_condition=None,
        next_factor_registered=False,GPU_hours=0,new_source_forward=0,verification_sha256=sha(R/'verification.json')))
    write(R/'closure_receipt.json',dict(files_sha256={str(p.relative_to(ROOT)):sha(p) for p in
        (R/'postresult_comparison.json',ROOT/'ROUND11_RESULTS.md',ROOT/'GOAL_TURN_R10_R11_AUDIT.json')},
        no_new_fit_or_statistical_selection=True))
    print(json.dumps(dict(primary_gain_pct=.9015169479602709,usage=[u for u in usage if u['view']=='core_real'],
        result_sha256=sha(R/'postresult_comparison.json'),report_sha256=sha(ROOT/'ROUND11_RESULTS.md')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

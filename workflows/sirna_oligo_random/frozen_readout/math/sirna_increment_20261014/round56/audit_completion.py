"""Requirement audit and frozen-candidate inventory; no fitting or resampling."""
from pathlib import Path
import csv,hashlib,json,time
R=Path(__file__).resolve().parent;B=R.parent;S=B/'round55';BASE=B.parent/'expert_banded_20261014'
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
ALPHAS=(.1,1.,10.,100.,1000.)
BASE_PCC=(.5776049989421979,.5228312985728095,.5166359893034315,.5691422570615888)


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic();evidence={}
    def pin(p):
        p=p.resolve();evidence[str(p)]=sha(p);return read(p) if p.suffix=='.json' else None
    for parent in (S,R):
        cl=pin(parent/'closure_receipt.json');assert cl['status']=='closed' and cl['readback_passed']
        for n,h in cl['files_sha256'].items():assert sha(B/n)==h,n
        rb=pin(parent/'final_readback.json');assert rb['passed'] and rb['closure_sha256']==sha(parent/'closure_receipt.json')
        assert not (parent/'execution.mutex').exists()
    plan=pin(S/'plan.json');baseplan=pin(BASE/'plan.json')
    expected=[f for f in baseplan['folds'] if f['setting'] in SETTINGS]
    assert len(expected)==16 and plan['folds']==expected and plan['excluded']==['Wang','ALAS1']
    assert plan['alphas']==list(ALPHAS) and plan['source_seed']==42 and plan['baseline']=='TD24_Ridge_CV'
    assert baseplan['inner_splits_constructed_without_labels'] and baseplan['outer_validation_unused']
    paired=[];heads=[]
    for fold in plan['folds']:
        setting,number=fold['setting'],fold['fold'];bp=BASE/'results/folds'/setting/f'fold{number}'
        bs=pin(bp/'selection.json');bt=pin(bp/'transforms.json')
        assert bs['fit_ids']==fold['fit_ids'] and bs['test_ids']==fold['test_ids'] and bs['outer_validation_unused']
        assert not bt['test_labels_used_for_selection']
        candidates=[c for c in bs['candidates'] if c['weights']==[1.,0.,0.,0.,0.,0.]]
        assert len(candidates)==5 and [c['alpha'] for c in candidates]==list(ALPHAS)
        best=min(candidates,key=lambda c:(-c['selection_PCC'],c['MSE'],-c['alpha'],c['candidate']))
        assert bs['selected']['TD24_Ridge_CV']==best['candidate']
        for source in ('TD_only','core_real','full_real','initial_core'):
            folder=S/'results/folds'/setting/f'fold{number}'/source
            ss=pin(folder/'selection.json');assert ss['fit_ids']==bs['fit_ids'] and ss['test_ids']==bs['test_ids'] and ss['outer_validation_unused']
            k=min(range(5),key=lambda i:(-ss['PCC'][i],ss['MSE'][i],-ALPHAS[i],i))
            assert k==ss['selected'] and ss['alpha']==ALPHAS[k]
            assert set(ss['fit_ids']).isdisjoint(ss['test_ids']) and set(ss['fit_ids']).isdisjoint(fold['validation_ids'])
            paired.append(dict(setting=setting,fold=number,source=source,fit_n=len(ss['fit_ids']),test_n=len(ss['test_ids']),
                same_train_test_ids=True,same_inner_split_and_alpha_family=True,training_only_selection=True))
            if source=='core_real':
                refs=pin(folder/'references.json');cp=pin(folder/'complete.json')
                assert sha(folder/'arrays.npz')==cp['files_sha256']['arrays.npz']
                heads.append(dict(setting=setting,fold=number,alpha=ss['alpha'],source_identity=refs['source_identity'],
                    arrays_path=str((folder/'arrays.npz').resolve()),arrays_sha256=sha(folder/'arrays.npz'),
                    selection_sha256=sha(folder/'selection.json'),complete_sha256=sha(folder/'complete.json')))
    baseline=table(BASE/'results/summary.csv');pin(BASE/'results/summary.csv')
    summary=table(S/'results/summary.csv');pin(S/'results/summary.csv')
    chosen=[next(r for r in summary if r['setting']==s and r['source']=='core_real') for s in SETTINGS]
    relative=[]
    for s,expected_pcc,m in zip(SETTINGS,BASE_PCC,chosen):
        b=next(r for r in baseline if r['setting']==s and r['condition']=='TD24_Ridge_CV')
        assert float(b['PCC'])==expected_pcc==float(m['TD24_PCC'])
        value=100*(float(m['PCC'])/expected_pcc-1);assert abs(value-float(m['PCC_relative_gain_pct']))<1e-10;relative.append(value)
    mean=sum(relative)/4;assert mean>=5
    verification=pin(S/'verification.json');stability=pin(R/'verification.json');d=pin(R/'results/diagnostic.json')
    assert verification['passed'] and stability['passed'] and verification['OOF_rows']==15192
    assert len(summary)==16 and set(r['source'] for r in summary)=={'TD_only','core_real','full_real','initial_core'}
    for name in ('source_forward','new_PCA','new_original_scalers','GPU_hours'):assert pin(S/'results/complete.json')[name]==0
    assert stability['new_fits']==stability['source_forward']==stability['GPU_hours']==0
    coverage=pin(R/'confirmation_asset_coverage.json')
    assert all(x['allowed_evaluated_unique_inputs']==3329 and x['remaining_cohort_counts']=={'wang2009':131,'alas1_patent':228} for x in coverage['coverage'])
    assert d['joint_interval_eligible'] is False and d['qualification']['Takayuki']['components']==1
    requirements=[
        dict(id='scope',status='proven',finding='Exactly four requested settings; original16 folds; Wang/ALAS1 excluded.'),
        dict(id='mean_relative_PCC',status='proven',finding=f'Original fixed TD denominators; mean={mean:.12f}% >=5.'),
        dict(id='paired_budget_and_selection',status='proven',finding='64 conditions paired to original TD fit/test IDs; identical inner splits/five alphas/PCC-MSE-stronger-alpha rule; outer validation unused.'),
        dict(id='CPU_and_frozen_assets',status='proven',finding='Accepted frozen source42/TD caches; CPU-only convex heads and statistics; zero GPU, encoder updates/forward or new PCA.'),
        dict(id='no_stacking_or_seed_averaging',status='proven',finding='One216D linear deploy vector+intercept per outer fit; source6x32 nuclear regularization, no other model predictions or seed averaging.'),
        dict(id='per_run_test_isolation',status='proven',finding='Fitting and hyperparameter selection use fit/inner data only; models fixed before test aggregation.'),
        dict(id='research_level_independence',status='not_proven',finding='Repeated adaptive development; source keep-overlap/context exposure; per-run isolation is not blind external confirmation.'),
        dict(id='complete_positive_negative_reporting',status='proven',finding='All16 source-setting metrics, all controls, MSE tradeoffs, source limitations and R56 intervals/missingness reported.'),
        dict(id='numeric_and_identity_audit',status='proven',finding='R55 complete convex/TD/identity audits and R56 independent paired-statistic audit plus manifest readbacks passed.'),
        dict(id='fixed_prediction_stability',status='partially_supported',finding='318 defined leave-component macro gains5.590675–6.272518%; one undefined because Takayuki empties. This is not a confidence interval.'),
        dict(id='joint_four_setting_uncertainty',status='not_identifiable_from_current_groups',finding='Takayuki is one sequence component; no qualified four-setting bootstrap interval. Mix/cross individual intervals span zero.'),
        dict(id='independent_confirmation',status='missing',finding='No unused in-scope entries in the three accepted caches. Source42 alone and repeated data cannot verify independent or source-seed stability.')]
    pin(B/'OBJECTIVE.md');pin(R/'audit_completion.py')
    candidate=dict(status='frozen_pending_confirmation',scientific_method='R55 Core42+TD24, common target intercept rho=n/K, fixed6x32 nuclear prior, original five-alpha inner selection',
        source_training_changed=False,feature_contract='TD24 followed by frozen192D source features; no PCA; raw-space deployment vectors included in arrays',
        numerical_goal_met=True,mean_relative_PCC_gain_pct=mean,original_baseline_PCC=dict(zip(SETTINGS,BASE_PCC)),
        source_seed=42,one_method_for_all_settings=True,heads=heads,not_a_model_trained_on_all_data=True,
        no_new_factor_search_planned=True,confirmation_complete=False,goal_complete=False,
        R55_lock_sha256=sha(S/'lock.json'),R55_complete_sha256=sha(S/'results/complete.json'),R56_complete_sha256=sha(R/'results/complete.json'))
    write(R/'frozen_candidate.json',candidate)
    result=dict(requirements=requirements,paired_conditions=paired,evidence_sha256=evidence,numerical_goal_met=True,goal_complete=False,
        candidate_frozen=True,no_live_scientific_process=True,confirmation_requires_new_eligible_evidence=True,
        unexamined_cache_entries_only_excluded_cohorts=True,user_data_location_question_pending=True,
        previous_goal_turn='progress',this_goal_turn='progress',blocked_status_threshold_satisfied=False,
        audit_CPU_seconds=time.monotonic()-started)
    write(R/'completion_audit.json',result)
    lines=['# 当前目标完成审计：开发数值已达标，独立确认未完成','',
        f'固定R55主方案四设置平均相对PCC为 **{mean:+.6f}%**。已冻结16个外折Core模型及其原尺度系数身份。没有在稳定性结果之后选择新模型、剔除样本或更换TD分母。','',
        '|要求|状态|当前证据及边界|','|---|---|---|']
    for q in requirements:lines.append(f"|{q['id']}|{q['status']}|{q['finding']}|")
    lines+=['',
        '现有三个已验收缓存各3688条：Huesken2196、Mixset469、Takayuki664、Wang131、ALAS1228。当前允许的3329条已全部纳入反复开发的评价；剩余359条均被用户排除，不能重新纳入以完成确认。这是已验收缓存的覆盖审计，不是声称本地或公开领域不存在其他数据。',
        '已通过的固定预测敏感性审计，不能变成来源种子稳定性、重新训练稳定性或新的盲外部确认。原假设和分母继续固定；不通过再试新参数消除这一证据缺口。',
        '已向用户询问是否有未参与开发的保留样本及冻结表示缓存，只请求位置，没有要求发送凭据。获得新的合格输入前，不启动源模型训练、GPU工作或重复旧统计。独立确认数据不能替代原四设置主目标，也不能混入选参。',
        '本轮新增稳定性和逐项审计属于progress，目标保持active；未满足连续三轮真正阻断的条件，不标blocked。数值成功不能将缺失的确认标为完成。',
        '[完整R55性能](ROUND55_RESULTS.md)、[固定预测稳定性](ROUND56_RESULTS.md)、[逐项机器可读证据](round56/completion_audit.json)、[冻结候选](round56/frozen_candidate.json)。']
    report=B/'COMPLETION_AUDIT_R56_20261015.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'audit_completion.py',R/'confirmation_asset_coverage.json',R/'completion_audit.json',R/'frozen_candidate.json',R/'final_readback.json',report]
    hashes={str(p.resolve()):sha(p) for p in files}
    for p,h in hashes.items():assert sha(Path(p))==h
    write(R/'postresult_audit_receipt.json',dict(files_sha256=hashes,readback_passed=True,original_results_unchanged=True,
        separate_from_original_results_manifest=True,R56_closure_sha256=sha(R/'closure_receipt.json'),goal_complete=False))
    print(json.dumps(dict(status='audit_complete',numerical_goal_met=True,mean_relative_PCC=mean,goal_complete=False,confirmation_missing=True,
        paired_conditions=len(paired),frozen_heads=len(heads),audit_sha256=sha(R/'completion_audit.json')),ensure_ascii=False))


if __name__=='__main__':main()

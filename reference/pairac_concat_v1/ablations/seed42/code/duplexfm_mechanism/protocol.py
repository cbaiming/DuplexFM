"""Fixed PairAC source protocol; efficiency search never selects by labels."""
SOURCE_SEEDS=(42,)
CONDITIONS=('no_pair','no_cross','no_access')
TRAIN_VARIANTS=tuple(f'{c}_seed{s}' for s in SOURCE_SEEDS for c in CONDITIONS)
ENGINEERING_VARIANTS=tuple(f'seed{s}' for s in SOURCE_SEEDS)+TRAIN_VARIANTS

def variant_parts(variant):
    if variant in tuple(f'seed{s}' for s in SOURCE_SEEDS):
        return 'full', int(variant[4:])
    if variant not in TRAIN_VARIANTS:
        raise ValueError('Unknown or unregistered ablation variant: '+str(variant))
    condition, seed=variant.rsplit('_seed',1)
    return condition,int(seed)
OFFICIAL_TESTS=('AGO2_CLASH_Hejret2023_test','AGO2_eCLIP_Klimentova2022_test',
                'AGO2_eCLIP_Manakov2022_test','AGO2_eCLIP_Manakov2022_leftout')

def source_config(assets):
    from pathlib import Path
    assets=Path(assets)
    return {'target_model':str(assets/'mrnabert'),'guide_model':str(assets/'rnafm'),
            'heads':8,'projection_dim':256,'hidden_width':32,'dropout':.1,
            'lora_rank':8,'lora_alpha':16,'lora_dropout':.05,'explicit_products':False,
            'local_accessibility':True,'site_energy':False,'accessibility_window':7,
            'architecture':'PairAC16','pair_embedding_dim':4}

def original_contract():
    return {'version':1,'architecture':'PairAC16','variants':list(TRAIN_VARIANTS),'source_seeds':list(SOURCE_SEEDS),
            'reference_epochs':8,'updates':32728,'reference_train_rows':2094107,'reference_val_rows':401428,
            'batch':512,'eval_batch':512,'amp':'bf16','learning_rate':3e-4,'lora_learning_rate':2e-5,
            'weight_decay':1e-4,'clip_grad_norm':1.,'checkpoint_selection':'fixed_terminal_update_only',
            'checkpoint_steps':500,'backup_seconds':3600,'max_gpu_hours':128,
            'pair_lookup':'16 ordered ACGU pairs x 4 learned coordinates; shared across positions',
            'input_channels':16,'manual_products':False,'BIO_branch':False,'Inta_branch':False,
            'accessibility':'existing independently frozen reactivity proxy; original and local difference',
            'positions':'guide absolute /32, target centered /30; 60-wide padded target, real guide length',
            'pooling':'existing guide regions 2-8,9-12,1+13-end; mean/max per candidate; no seed-match gate',
            'transfer_interface':'source192 at explicitly provided siRNA design site; no energy',
            'source_protocol':'existing keep-overlap; unchanged row partitions',
            'source_tests':list(OFFICIAL_TESTS),'test_selection':False,'source_seed_averaging':'seed42 only; between-source-seed SD unavailable',
            'initialization':'public foundations plus new task LoRA and pair lookup; exact pre-update state saved'}

def contract():
    value=original_contract()
    value.update(version=3,experiment='PairAC_fixed_mask_expert_ablation',conditions=list(CONDITIONS),
        micro_batch=512,full_reference='completed PairAC seed42; original initial reused exactly',
        ablation='fixed input-channel zero mask on every train/eval/source192 forward; no stochastic routing',
        masks={'no_pair':[0,1,2,3],'no_cross':list(range(4,12)),'no_access':[12,13]},
        positions_retained=[14,15],nominal_task_parameters=1198218,parameter_shapes_preserved=True,
        effective_capacity_equal=False,masked_owned_parameters='optimizer eligible but gradients set None; no AdamW decay',
        efficiency_search='CPU threads1/2/4 at fixed original micro/global batch512; no LR or model selection',
        transfer='source192 exported later with the SAME condition mask; fixed R55 TD24+source192 readout, no retuning',
        interpretation='conditional information necessity under retraining; not biological causality or absence of all sequence knowledge')
    return value

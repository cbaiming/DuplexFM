"""Fixed PairAC source protocol; efficiency search never selects by labels."""
TRAIN_VARIANTS=('seed42','seed43','seed44')
ENGINEERING_VARIANTS=TRAIN_VARIANTS
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

def contract():
    return {'version':1,'architecture':'PairAC16','variants':list(TRAIN_VARIANTS),'source_seeds':[42,43,44],
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
            'source_tests':list(OFFICIAL_TESTS),'test_selection':False,'source_seed_averaging':'metrics mean/SD; no ensemble as main result',
            'initialization':'public foundations plus new task LoRA and pair lookup; exact pre-update state saved'}

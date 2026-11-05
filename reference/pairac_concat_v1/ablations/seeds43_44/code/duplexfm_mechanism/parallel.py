"""Two independent H100 workers, durable session accounting, audited recovery."""
from __future__ import annotations

from contextlib import ExitStack
import datetime as dt
import hashlib
import os
from pathlib import Path
import platform
import queue
import signal
import subprocess
import sys
import threading
import time

from .common import FileLock, backup, check_disk, read, sha, verify_files
from .durable_io import durable_json as write, independent_copy, recover_checkpoint, valid_checkpoint
from .protocol import TRAIN_VARIANTS as VARIANTS, contract, source_config, variant_parts
from .runtime import choose_gpu, dispatch_gpu, gpu_inventory, thread_env, verify_registration

EXECUTION = {
    'version': 3, 'cards': 2, 'experiment_allocation_gpu_hours': 128,
    'platform_total_gpu_hours': 128, 'backup_seconds': 3600,
    'heartbeat_seconds': 30, 'checkpoint_steps': 500, 'cpu_threads_per_worker': 4,
    'dispatch_order': list(VARIANTS), 'worker_mode': 'independent_single_GPU',
    'efficiency_search': {'micro_batches':[512], 'cpu_threads':[1,2,4],
                          'warmup_updates':2, 'measured_updates':6, 'max_wall_seconds':3600,
                          'reserved_memory_fraction':.85, 'near_fastest_fraction':.97,
                          'global_batch':512, 'learning_rate_search':False},
    'CPU_allocation': 'min(OS CPU count, affinity, cgroup quota); require 2*worker_threads+2<=vCPUs',
    'overrides': 'User-authorized 128 allocated GPU hours and hourly backup replace upstream operational 48 hours/7200 seconds; scientific training stays fixed. The legacy single-card entry is disabled.',
    'accounting': 'both allocated cards times the entire command session, including idle/CPU/failure/save',
    'abrupt_recovery': 'exclusive locks plus operator-provided platform termination time; do not guess cost',
    'model': 'PairAC fixed single-input masks; 3 conditions x remaining seeds43/44; same nominal132 tensors/1198218parameters; exact original Full initial',
    'H100_preflight': 'Full and all masks at both registered seeds BF16 forward/backward before CPU-thread calibration; engineering updates discarded',
}
CARDS = EXECUTION['cards']
JOBS = VARIANTS + ('evaluation',)


def signal_worker(process, sig):
    """Signal only a child owned by this controller; support CPU tests on Windows."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        (process.terminate if sig == signal.SIGTERM else process.kill)()
    else:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            if process.poll() is None:
                raise


def register(root, assets):
    root = Path(root).resolve()
    old = verify_registration(root, assets)
    if (root / 'parallel_ledger.json').exists():
        raise FileExistsError('Parallel execution already registered; reuse it')
    if read(root / 'ledger.json') != {'active': None, 'limit_seconds': 172800, 'phases': []}:
        raise ValueError('Only a never-trained new execution root can be registered')
    write(root / 'parallel_execution.json', EXECUTION, immutable=True)
    write(root / 'parallel_registration.json', {
        'source_registration_sha256': sha(root / 'registration.json'),
        'execution_sha256': sha(root / 'parallel_execution.json'),
        'source_contract': old['contract'],
    }, immutable=True)
    write(root / 'parallel_ledger.json', {'version': 1, 'sessions': [], 'active': None,
                                       'experiment_limit_gpu_seconds': EXECUTION['experiment_allocation_gpu_hours'] * 3600,
                                       'platform_limit_gpu_seconds': 128 * 3600}, immutable=True)
    return {'status': 'parallel_registered', 'cards': CARDS, 'backup_seconds': 3600}


def verify(root, assets):
    root = Path(root)
    source = verify_registration(root, assets)
    ready=read(root/'environment/ready.json')
    verify_files(root,ready['files_sha256'])
    if ready['status']!='passed':raise ValueError('Preparation did not pass')
    execution = read(root / 'parallel_execution.json')
    frozen = read(root / 'parallel_registration.json')
    if (execution != EXECUTION or frozen['source_registration_sha256'] != sha(root / 'registration.json')
            or frozen['execution_sha256'] != sha(root / 'parallel_execution.json')
            or frozen['source_contract'] != source['contract']):
        raise ValueError('Parallel execution/source/environment lock changed')
    return source


def select_cards(requested, inventory):
    names = [v.strip() for v in requested.split(',') if v.strip()]
    if len(names) != CARDS or len(set(names)) != CARDS:
        raise ValueError('Exactly two different requested GPU indices/UUIDs are required')
    gpus, processes = inventory
    selected = [choose_gpu(gpus, processes, name, 'H100') for name in names]
    if len({g['uuid'] for g in selected}) != CARDS:
        raise ValueError('GPU aliases resolve to duplicate physical cards')
    if len({(g['name'],g['memory_mib']) for g in selected}) != 1:
        raise ValueError('Shared efficiency calibration requires two matching H100 models/memory sizes')
    return selected


def charged_seconds(ledger):
    if ledger['active'] is not None:
        raise RuntimeError('Previous live session requires explicit audited recovery')
    if (ledger['experiment_limit_gpu_seconds'] != EXECUTION['experiment_allocation_gpu_hours'] * 3600
            or ledger['platform_limit_gpu_seconds'] != 128 * 3600):
        raise ValueError('Allocation budget lock differs')
    used = sum(s['allocation_gpu_seconds'] for s in ledger['sessions'])
    if used < 0:
        raise ValueError('Negative session cost')
    return used


def remaining_wall(ledger):
    used = charged_seconds(ledger)
    return max(0., min(ledger['experiment_limit_gpu_seconds'], ledger['platform_limit_gpu_seconds']) - used) / CARDS


def process_identity(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return None
    # The parenthesized comm may contain whitespace.
    tail = path.read_text().rpartition(')')[2].split()
    return {'pid': pid, 'start_ticks': tail[19], 'hostname': platform.node(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def check_complete(root, variant):
    if variant == 'evaluation':
        from .evaluation import check_complete as evaluated
        return evaluated(root)
    folder = Path(root) / 'runs' / variant
    if not (folder / 'complete.json').exists():
        return False
    value = read(folder / 'complete.json')
    if value['updates'] != 32728 or value['selection'] != 'terminal_update' or value['reference_epochs'] != 8:
        raise ValueError('Partial/best-validation output cannot be promoted')
    for file, field in [('initial.pt', 'initial_sha256'), ('terminal.pt', 'terminal_sha256'),
                        ('signature.json', 'signature_sha256'), ('history.json', 'history_sha256')]:
        if sha(folder / file) != value[field]:
            raise ValueError(f'Completed {variant} artifact differs: {file}')
    expected = read(folder / 'signature.json')
    initial = valid_checkpoint(folder / 'initial.pt', expected)
    terminal = valid_checkpoint(folder / 'terminal.pt', expected)
    if initial['state']['step'] != 0 or terminal['state']['step'] != 32728:
        raise ValueError('Completed checkpoint cursor differs')
    return True


def recovery_time(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        parsed = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('Use an epoch or an ISO8601 timestamp with timezone')
        result = parsed.timestamp()
    return result


def recover(root, assets, stopped_at):
    """No process is killed and no unchecked live ledger is silently cleared."""
    root = Path(root).resolve()
    with FileLock(root / 'parallel_controller.lock'):
        verify(root, assets)
        ledger = read(root / 'parallel_ledger.json')
        active = ledger['active']
        if active is None:
            return {'status': 'already_closed', 'charged_gpu_seconds': charged_seconds(ledger)}
        stopped = recovery_time(stopped_at)
        if not active['last_heartbeat_unix'] <= stopped <= time.time() + 5:
            raise ValueError('Platform termination time must follow last recorded heartbeat and not be in the future')
        # Distributed file locks cover workers left alive after controller death.
        # Same-host PID start-time checks additionally avoid PID-reuse mistakes.
        with ExitStack() as stack:
            for variant in ('efficiency',) + JOBS:
                stack.enter_context(FileLock(root / 'worker_locks' / f'{variant}.lock'))
            for item in [active['controller_identity']] + [w.get('identity') for w in active['workers'] if w.get('identity')]:
                current = process_identity(item['pid'])
                if current == item:
                    raise RuntimeError('Recorded controller/worker is still alive; recovery refused')
            receipt_root = root / 'recovery_receipts' / str(time.time_ns())
            receipt_root.mkdir(parents=True)
            independent_copy(root / 'parallel_ledger.json', receipt_root / 'ledger_before.json')
            restored = {}
            for variant in VARIANTS:
                folder = root / 'runs' / variant
                if check_complete(root, variant):
                    restored[variant] = {'status': 'completed_reused'}
                elif (folder / 'signature.json').exists():
                    restored[variant] = recover_checkpoint(folder)
            active.update(ended_unix=stopped, allocation_gpu_seconds=CARDS * (stopped - active['started_unix']),
                          status='recovered_abrupt_stop', cost_basis='operator_platform_termination_time',
                          recovery_receipt=str(receipt_root.relative_to(root)))
            ledger['sessions'].append(active)
            ledger['active'] = None
            write(receipt_root / 'audit.json', {'platform_stopped_at': stopped, 'restored': restored,
                  'scope': 'resume saved cursor; unsaved updates and interrupted validation are replayed'}, immutable=True)
            write(root / 'parallel_ledger.json', ledger)
        return {'status': 'recovered', 'restored': restored, 'remaining_wall_seconds': remaining_wall(ledger)}


def worker_command(root, assets, lease):
    return [sys.executable, '-u', str(Path(root) / 'code/parallel_run.py'), 'worker',
            '--root', str(root), '--assets', str(assets), '--lease', str(lease)]


def controller(root, assets, requested='0,1'):
    command_started=time.time()
    try:
        return _controller(root, assets, requested, command_started)
    except BaseException as exc:
        # Failures before child dispatch still consume allocated platform time.
        # Never clear a pre-existing live session or a corrupted cost ledger.
        root=Path(root).resolve()
        with FileLock(root/'parallel_controller.lock'):
            value=read(root/'parallel_ledger.json')
            if value['active'] is None and not any(s['started_unix']>=command_started for s in value['sessions']):
                charged_seconds(value)
                ended=time.time()
                row={'id':str(time.time_ns()),'allocated_cards':CARDS,'started_unix':command_started,
                     'ended_unix':ended,'allocation_gpu_seconds':CARDS*(ended-command_started),
                     'status':'preflight_failure','workers':[], 'error':f'{type(exc).__name__}: {exc}'}
                value['sessions'].append(row)
                write(root/'parallel_ledger.json',value)
        raise


def _controller(root, assets, requested, command_started):
    root, assets = Path(root).resolve(), Path(assets).resolve()
    with FileLock(root / 'parallel_controller.lock'):
        verify(root, assets)
        ledger = read(root / 'parallel_ledger.json')
        wall = remaining_wall(ledger)
        completed = [v for v in JOBS if check_complete(root, v)]
        if len(completed) == len(JOBS):
            return {'status': 'source_training_complete', 'completed': completed,
                    'charged_allocation_gpu_seconds': charged_seconds(ledger)}
        if wall < 600:
            return {'status': 'budget_exhausted', 'completed': completed, 'remaining_wall_seconds': wall}
        check_disk(root)
        selected = select_cards(requested, gpu_inventory())
        from .efficiency import load_plan
        calibration_needed = not (root/'calibration/complete.json').exists()
        if not calibration_needed:
            plan = load_plan(root)
            if any((g['name'],g['memory_mib']) != (plan['hardware_name'],plan['memory_mib']) for g in selected):
                raise ValueError('Saved hardware plan does not match the new H100 task')
        with ExitStack() as stack:
            for g in selected:
                stack.enter_context(FileLock(Path('/tmp') / ('duplexfm_' + g['uuid'] + '.lock')))
            select_cards(requested, gpu_inventory())
            started = command_started
            session = {'id': str(time.time_ns()), 'cards': selected, 'allocated_cards': CARDS,
                       'started_unix': started, 'deadline_unix': started + wall,
                       'last_heartbeat_unix': started, 'controller_identity': process_identity(os.getpid()),
                       'workers': [], 'status': 'active', 'execution_sha256': sha(root / 'parallel_execution.json')}
            ledger['active'] = session
            write(root / 'parallel_ledger.json', ledger)
            events = queue.Queue()
            old_signals = {}
            def stopping(signum, _frame):
                events.put(('stop', signum))
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, stopping)
            pending = ['efficiency'] if calibration_needed else [v for v in JOBS if v not in completed]
            running, previous_exit, completed_durations = {}, {}, []
            stop_reason, grace_deadline = None, None
            latest_ledger_backup = started

            def seal():
                session['last_heartbeat_unix'] = time.time()
                session['live_allocation_gpu_seconds'] = CARDS * (time.time() - started)
                write(root / 'parallel_ledger.json', ledger)

            def launch(variant, gpu):
                nonlocal pending
                folder = root / 'runs' / variant
                with FileLock(root / 'worker_locks' / f'{variant}.lock'):
                    if (folder / 'signature.json').exists():
                        recover_checkpoint(folder)
                confirmed = dispatch_gpu(gpu['uuid'], previous_exit.get(gpu['uuid']))
                lease_name = f'leases/{session["id"]}_{variant}.json'
                task_deadline = min(session['deadline_unix'],time.time()+3600) if variant=='efficiency' else session['deadline_unix']
                lease = {'variant': variant, 'gpu': confirmed, 'session_id': session['id'],
                         'deadline_unix': task_deadline, 'execution_sha256': session['execution_sha256'],
                         'registration_sha256': sha(root / 'registration.json'), 'lease_path': lease_name}
                write(root / lease_name, lease, immutable=True)
                row = {'variant': variant, 'gpu': confirmed, 'lease_path': lease_name,
                       'started_unix': time.time(), 'status': 'launching'}
                session['workers'].append(row)
                seal()  # Durable intent before process creation.
                logs = root / 'logs'; logs.mkdir(exist_ok=True)
                env = thread_env(4 if variant=='efficiency' else load_plan(root)['cpu_threads']); env['CUDA_VISIBLE_DEVICES'] = confirmed['uuid']
                with (logs / f'{session["id"]}_{variant}.log').open('xb') as log:
                    process = subprocess.Popen(worker_command(root, assets, root / lease_name), env=env,
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=root)
                row.update(pid=process.pid, identity=process_identity(process.pid), status='running')
                running[confirmed['uuid']] = (process, row)
                seal()
                def wait_child():
                    events.put(('exit', confirmed['uuid'], process.wait()))
                threading.Thread(target=wait_child, daemon=True).start()

            try:
                while pending or running:
                    if (root / 'STOP_REQUESTED').exists() and stop_reason is None:
                        stop_reason = 'operator_stop_file'
                    if stop_reason is None:
                        available = session['deadline_unix'] - time.time() - 300
                        for g in selected:
                            if g['uuid'] in running or not pending:
                                continue
                            predicted = 3600 if pending[0]=='efficiency' else max(completed_durations, default=8 * 3600) * 1.2
                            next_variant = pending[0]
                            if next_variant == 'evaluation' and not all(v in completed for v in VARIANTS):
                                continue
                            resumed = (root / 'runs' / next_variant / 'last.pt').exists()
                            if not resumed and available < predicted:
                                continue
                            launch(pending.pop(0), g)
                        if not running:
                            stop_reason = 'budget_reserve_stop'
                            break
                    remaining = session['deadline_unix'] - time.time()
                    if remaining <= 300 and stop_reason is None:
                        stop_reason = 'budget_deadline'
                    if stop_reason is not None and grace_deadline is None:
                        grace_deadline = time.time() + 180
                        for process, _ in running.values():
                            if process.poll() is None:
                                signal_worker(process, signal.SIGTERM)
                    if stop_reason is not None and not running:
                        break
                    if grace_deadline is not None and time.time() >= grace_deadline:
                        for process, _ in running.values():
                            if process.poll() is None:
                                signal_worker(process, signal.SIGKILL)
                    try:
                        timeout = min(30., max(.1, remaining - 300)) if stop_reason is None else 30.
                        event = events.get(timeout=timeout)
                    except queue.Empty:
                        event = ('heartbeat',)
                    if event[0] == 'stop':
                        stop_reason = 'operator_signal'
                    elif event[0] == 'exit':
                        _, uuid, rc = event
                        process, row = running.pop(uuid)
                        good = (root/'calibration/complete.json').exists() if row['variant']=='efficiency' else check_complete(root,row['variant'])
                        row.update(returncode=rc, ended_unix=time.time(), status='complete' if rc == 0 and good else 'interrupted_or_failed')
                        if row['status'] == 'complete':
                            if row['variant']=='efficiency':
                                load_plan(root)
                                pending = [v for v in JOBS if v not in completed]
                            else:
                                completed.append(row['variant'])
                                completed_durations.append(row['ended_unix'] - row['started_unix'])
                            previous_exit[uuid] = time.monotonic()
                        else:
                            stop_reason = stop_reason or 'worker_failure_or_interruption'
                    seal()
                    if time.time() - latest_ledger_backup >= 3600:
                        backup(root, [root / 'parallel_ledger.json', root / 'parallel_registration.json',
                                      root / 'parallel_execution.json'], 'parallel_hourly_cost')
                        latest_ledger_backup = time.time()
                status = 'source_training_complete' if len(completed) == len(JOBS) else (stop_reason or 'stopped')
            except BaseException:
                status = 'controller_exception'
                for process, _ in running.values():
                    if process.poll() is None:
                        signal_worker(process, signal.SIGTERM)
                for process, row in running.values():
                    try:
                        rc = process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        signal_worker(process, signal.SIGKILL)
                        rc = process.wait(timeout=30)
                    row.update(returncode=rc, ended_unix=time.time(), status='exception_cleanup')
                status = 'controller_exception'
                raise
            finally:
                for sig, handler in old_signals.items():
                    signal.signal(sig, handler)
                # Final backup is included in the allocated-card session clock.
                try:
                    seal()
                    backup(root, [root / 'parallel_ledger.json', root / 'parallel_execution.json',
                                  root / 'parallel_registration.json'], 'parallel_closing_cost')
                finally:
                    ended=time.time()
                    session.update(ended_unix=ended, allocation_gpu_seconds=CARDS * (ended - started), status=status)
                    ledger['sessions'].append(session.copy()); ledger['active'] = None
                    write(root / 'parallel_ledger.json', ledger)
            result = {'status': status, 'completed': completed, 'pending': [v for v in JOBS if v not in completed],
                      'charged_allocation_gpu_seconds': charged_seconds(ledger),
                      'parallel_ledger_sha256': sha(root / 'parallel_ledger.json'), 'evaluation_pending': not check_complete(root,'evaluation')}
            write(root / 'parallel_status.json', result)
            if status == 'source_training_complete':
                write(root / 'source_training_complete.json', result, immutable=True)
            return result


def worker(root, assets, lease_file):
    root, assets = Path(root).resolve(), Path(assets).resolve()
    lease_file = Path(lease_file).resolve()
    if not lease_file.is_relative_to(root / 'leases'):
        raise ValueError('Worker lease leaves execution root')
    lease = read(lease_file)
    variant = lease['variant']
    if variant not in ('efficiency',) + JOBS or os.environ.get('CUDA_VISIBLE_DEVICES') != lease['gpu']['uuid']:
        raise ValueError('Worker may see exactly its assigned GPU')
    with FileLock(root / 'worker_locks' / f'{variant}.lock'):
        import numpy as np
        import torch
        from duplexfm_joint.data import PairStore
        from duplexfm_joint.io_utils import provenance_content, seed_all
        from .model import PairACModel
        from .training_hourly import StopGuard, run_engine
        registered = verify(root, assets)
        active = read(root / 'parallel_ledger.json')['active']
        if (not active or active['id'] != lease['session_id']
                or sha(root / 'parallel_execution.json') != lease['execution_sha256']
                or sha(root / 'registration.json') != lease['registration_sha256']):
            raise ValueError('Worker controller lease is no longer current')
        if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
            raise ValueError('Formal worker requires exactly one H100')
        guard = StopGuard(lease['deadline_unix'] - (60 if variant=='efficiency' else 300), root / 'STOP_REQUESTED')
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(guard, 'signal_received', True))
        if variant=='efficiency':
            from .efficiency import calibrate
            result = calibrate(root,assets,lease,guard)
            write(root/'leases'/(lease_file.stem+'.worker_end.json'),
                  {'variant':variant,'session_id':lease['session_id'],'result':result,'ended_unix':time.time(),
                   'identity':process_identity(os.getpid())},immutable=True)
            return result
        if variant=='evaluation':
            from .evaluation import evaluate
            return evaluate(root,assets,guard)
        from .efficiency import load_plan
        plan=load_plan(root)
        condition,seed=variant_parts(variant)
        seed_all(seed); torch.set_num_threads(plan['cpu_threads'])
        with PairStore(assets / 'mirbench') as store:
            integrity = store.verify_integrity(); prep = root / 'preparation'
            fit_file = prep / 'train.npy'
            fit, val = np.load(fit_file, allow_pickle=False), np.load(prep / 'validation.npy', allow_pickle=False)
            model = PairACModel(source_config(assets), variant).to('cuda:0')
            from .reference_initial import enforce_reference_initial
            initial_proof=enforce_reference_initial(model,root,seed)
            spec = contract()
            options = {'seed': seed, 'reference_rows': spec['reference_train_rows'], 'batch_size': spec['batch'],
                       'eval_batch_size': spec['eval_batch'], 'max_updates': spec['updates'], 'amp': 'bf16',
                       'learning_rate': spec['learning_rate'], 'lora_learning_rate': spec['lora_learning_rate'],
                       'weight_decay': spec['weight_decay'], 'clip_grad_norm': spec['clip_grad_norm'],
                       'checkpoint_steps': 500, 'backup_seconds': 3600, 'micro_batch_size':plan['micro_batch']}
            signature = {'variant': variant, 'source_seed': seed, 'condition':condition, 'full_initial_proof':initial_proof, 'registration_sha256': sha(root / 'registration.json'),
                         'parallel_registration_sha256': sha(root / 'parallel_registration.json'),
                         'efficiency_plan_sha256':sha(root/'efficiency_plan.json'),
                         'foundation': provenance_content(model.base_provenance), 'fit_sha256': sha(fit_file),
                         'validation_sha256': sha(prep / 'validation.npy'), 'data_manifest_sha256': integrity['manifest_sha256'],
                         'model_config': {k:v for k,v in source_config(assets).items() if not k.endswith('_model')}}
            # A lease is an execution identity, not part of the resumable optimizer signature.
            shuffled = None
            folder = root / 'runs' / variant
            result = run_engine(model, store, fit, val, folder, options, signature,
                                resume=(folder / 'last.pt').exists(), shuffled_labels=shuffled, backup_root=root, guard=guard,
                                backup_extras=[lease_file])
        write(root / 'leases' / (lease_file.stem + '.worker_end.json'), {
            'variant': variant, 'session_id': lease['session_id'], 'result': result,
            'ended_unix': time.time(), 'identity': process_identity(os.getpid()),
        }, immutable=True)
        return result

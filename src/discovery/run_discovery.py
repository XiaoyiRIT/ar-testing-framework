# discovery/run_discovery.py
import json, time, pathlib
from policy.policy import NMPolicy
from verifier.verifier import Verifier

def run_discovery_once(drv, detector, sampler, executor, verifier: Verifier, policy: NMPolicy, cfg):
    # 1) 检测
    det = detector.detect()
    targets = select_targets(det)  # 选择若干 AR 物体与若干非AR区域

    results = []
    for region in targets:
        for op_type in ['tap','drag','rotate']:
            trial_success = []
            for i in range(policy.N):
                params = policy.sample_params(op_type)
                before = snapshot_screen(drv)
                # 2) 执行
                executor.perform(op_type, region, params)
                time.sleep(cfg['post_wait_s'])
                after  = snapshot_screen(drv)

                # 3) 重新检测（或用跟踪）得到 det_after
                det_after = detector.detect()

                # 4) 验证
                succ, evidence, metrics = verifier.verify(
                    op_type, before, after, det, det_after, region['target_id'], extra={}
                )
                trial_success.append(succ)

                # 5) 落盘一次 trial
                rec = pack_jsonl_record(op_type, region, params, succ, metrics, evidence, cfg)
                append_jsonl(cfg['out_jsonl'], rec)

            # 6) N/M 判定 → 写入“支持矩阵”条目
            support = policy.decide_support(op_type, trial_success)
            save_support(cfg['support_jsonl'], region, op_type, support, trial_success)

    return True

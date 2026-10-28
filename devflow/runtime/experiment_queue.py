"""Run predeclared experiments serially after verifying immutable inputs."""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from .artifacts import atomic_json, sha256


def refresh_monitor(output_dir):
    """Best-effort charts must never gate scientific completion or later stages."""
    output=Path(output_dir)
    error_path=output/"MONITOR_REFRESH_ERROR.json"
    try:
        from .monitor import build_monitor
        build_monitor(output)
        error_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    status="error",error=repr(exc),traceback=traceback.format_exc())
        try:
            atomic_json(error_path,record)
        except Exception as logging_error:
            # Diagnostics are optional even when their own error file cannot be
            # written. The queue's required validation records remain strict.
            try:
                print(json.dumps(dict(monitor_refresh_error=record,
                                      monitor_error_logging_error=repr(logging_error))),
                      file=sys.stderr,flush=True)
            except Exception:
                pass
        return False


def run(plan_path):
    plan_path=Path(plan_path).resolve()
    plan=json.loads(plan_path.read_text())
    root=Path(plan["run_root"])
    root.mkdir(parents=True,exist_ok=True)
    def state(**kwargs):
        atomic_json(root/"queue_status.json",dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kwargs))
    def paused():return (root/"USER_PAUSED.json").exists()
    def verify():
        for name,expected in plan["source_sha256"].items():
            if sha256(Path(plan["source_root"])/name)!=expected:raise RuntimeError("Source changed: "+name)
    with (root/".queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        process=None
        try:
            for stage in plan["stages"]:
                if paused():state(status="paused_by_user");return
                verify()
                if sha256(stage["config"])!=stage["config_sha256"]:raise RuntimeError("Stage configuration changed")
                while True:
                    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader"],text=True).strip()
                    if not apps:break
                    state(status="waiting_for_gpu",stage=stage["name"],gpu_processes=apps)
                    if paused():state(status="paused_by_user");return
                    time.sleep(20)
                command=[plan["python"],"-m","torch.distributed.run","--standalone","--nproc_per_node=8","-m","recipes.sciplex3","--config",stage["config"],"train"]
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7","PYTHONPATH":plan["source_root"],"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
                log=root/"logs"/(stage["name"]+".log");log.parent.mkdir(exist_ok=True)
                with log.open("ab",buffering=0) as stream:
                    process=subprocess.Popen(command,cwd=plan["source_root"],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    state(status="running",stage=stage["name"],child_pid=process.pid,log=str(log),source_commit=plan.get("source_commit"))
                    last_monitor=0.
                    while process.poll() is None:
                        if paused():
                            os.killpg(process.pid,signal.SIGTERM)
                            process.wait(timeout=60)
                            state(status="paused_by_user",stage=stage["name"]);return
                        if time.monotonic()-last_monitor>60:
                            refresh_monitor(stage["output_dir"])
                            last_monitor=time.monotonic()
                        observed={}
                        observed_path=Path(stage["output_dir"])/"status.json"
                        if observed_path.exists():observed=json.loads(observed_path.read_text())
                        current_update=observed.get("update")
                        phase="evaluating" if current_update==stage["expected_updates"] and observed.get("status")=="training" else "training"
                        state(status="running",phase=phase,stage=stage["name"],child_pid=process.pid,update=current_update,log=str(log),source_commit=plan.get("source_commit"))
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:pass
                if process.returncode:raise RuntimeError(stage["name"]+" failed; see "+str(log))
                output=Path(stage["output_dir"])
                completed=json.loads((output/"TRAINING_COMPLETE.json").read_text())
                first=json.loads((output/"FIRST_UPDATE_VERIFIED.json").read_text())
                if completed.get("status")!="complete" or completed.get("updates")!=stage["expected_updates"] or first.get("status")!="passed":raise RuntimeError("Completion validation failed")
                summary=output/"evaluations"/f"update_{stage['expected_updates']:06d}"/"summary.json"
                if not summary.is_file() or json.loads(summary.read_text()).get("n_conditions")!=108:raise RuntimeError("Final online evaluation missing")
                refresh_monitor(output)
                atomic_json(output/"QUEUE_STAGE_VERIFIED.json",dict(status="passed",updates=stage["expected_updates"],summary_sha256=sha256(summary)))
                state(status="stage_complete",stage=stage["name"])
            atomic_json(root/"QUEUE_COMPLETED.json",dict(status="complete",stages=len(plan["stages"]),plan_sha256=sha256(plan_path)))
            state(status="complete",stages=len(plan["stages"]))
        except BaseException as exc:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
            state(status="failed",error=repr(exc),traceback=traceback.format_exc())
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    run(parser.parse_args().plan)

if __name__=="__main__":main()

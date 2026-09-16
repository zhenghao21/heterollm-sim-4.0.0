"""Elevated, bounded clock lifecycle for immutable synthetic matrix; never kills children."""
import ctypes,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;C=P/'collection_r2';PY=Path(r'E:\anaconda\python.exe');SMI=Path(r'C:\Windows\System32\nvidia-smi.exe')
SHA='d78e2713ef56fd5e73b344452b27a0b5af4601ac2a1b6bf05d38ae41fce55529'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,d):
 with path.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2);f.write('\n')
def ref(path):
 b=path.read_bytes();return {'path':str(path.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def capture(argv,name):
 with (P/(name+'.stdout.txt')).open('xb') as out,(P/(name+'.stderr.txt')).open('xb') as err:
  r=subprocess.run(argv,stdout=out,stderr=err,creationflags=FLAGS)
 return {'command':list(map(str,argv)),'returncode':r.returncode,'stdout_ref':ref(P/(name+'.stdout.txt')),'stderr_ref':ref(P/(name+'.stderr.txt'))}
def wait_owned():
 # A checkpoint is not exit. Keep the hardware domain until every launched worker exits.
 for directory in (C/'runs').glob('*/*/*'):
  supervisor=directory/'supervisor.json'
  if supervisor.exists() and not (directory/'complete.json').exists():
   pid=json.loads(supervisor.read_text())['pid'];k=ctypes.WinDLL('kernel32',use_last_error=True)
   k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
   h=k.OpenProcess(0x100000,False,pid)
   if h:
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p]
    try:
     while k.WaitForSingleObject(h,1000)==258:pass
    finally:k.CloseHandle(h)
def main():
 locked=False;result={'schema':'operator-clock-lifecycle/v1','started_utc':utc(),'pid':os.getpid(),'controller_ref':ref(Path(__file__))}
 try:
  if not ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('elevated token required')
  if ref(C/'freeze.json')['sha256']!=SHA:raise RuntimeError('approved collection changed')
  capture([str(SMI),'--query-gpu=uuid,pstate,clocks.sm,clocks.mem','--format=csv,noheader'],'clock_before')
  lock=capture([str(SMI),'-lgc','2400,2400'],'clock_lock')
  if lock['returncode']!=0:raise RuntimeError('clock lock rejected: see captured output')
  locked=True;time.sleep(1)
  check=capture([str(SMI),'--query-gpu=uuid,clocks.sm','--format=csv,noheader,nounits'],'clock_readback')
  fields=(P/'clock_readback.stdout.txt').read_text(encoding='utf-8-sig').strip().split(',')
  if check['returncode']!=0 or len(fields)!=2 or abs(int(fields[1].strip())-2400)>30:raise RuntimeError('actual readback does not match 2400MHz domain')
  start=P/'clock_control_start.json'
  write(start,{'schema':'operator-clock-control-receipt/v1','created_utc':utc(),'gpu_uuid':fields[0].strip(),'target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,'requested_lock_min_mhz':2400,'requested_lock_max_mhz':2400,'lock_command_returncode':lock['returncode'],'command':lock['command'],'stdout_ref':lock['stdout_ref'],'stderr_ref':lock['stderr_ref'],'restore_on_exit_planned':True,'readback':check,'controller_ref':ref(Path(__file__))})
  args=[str(PY),str(C/'runner.py'),'run','--root-reviewed','--idle-window-confirmed','--expected-freeze-sha256',SHA,'--clock-control-receipt',str(start),'--clock-control-sha256',ref(start)['sha256']]
  first=capture(args+['--max-stages','3'],'collection_first_pair');wait_owned();result['first_pair']=first
  receipts=list((C/'runs').glob('*/*/*/complete.json'))
  statuses=[json.loads(r.read_text(encoding='utf-8'))['status'] for r in receipts]
  if len(statuses)!=3 or any(x!='completed' for x in statuses):raise RuntimeError('first pair did not complete successfully; retain failures and repair in new freeze')
  write(P/'first_pair_ready.json',{'utc':utc(),'receipts':[ref(r) for r in receipts],'controller_pid':os.getpid(),'continuation_file':str(P/'continue_collection.json')})
  # Root inspects actual schema before continuing; keep one unchanged clock session.
  deadline=time.monotonic()+600
  while not (P/'continue_collection.json').exists():
   if (P/'stop_collection.json').exists():raise RuntimeError('root requested stop after finished pair')
   if time.monotonic()>deadline:raise RuntimeError('root review window expired before continuation')
   time.sleep(1)
  consent=json.loads((P/'continue_collection.json').read_text(encoding='utf-8'))
  if consent.get('approved_freeze_sha256')!=SHA or consent.get('first_pair_reviewed') is not True:raise RuntimeError('continuation identity missing')
  result['remaining_matrix']=capture(args,'collection_remaining_matrix');wait_owned();result['status']='finished'
 except BaseException as e:result.update(status='failed',error=type(e).__name__+': '+str(e))
 finally:
  wait_owned()
  if locked:result['clock_reset']=capture([str(SMI),'-rgc'],'clock_reset')
  result['finished_utc']=utc();write(P/'clock_control_finish.json',result)
if __name__=='__main__':main()

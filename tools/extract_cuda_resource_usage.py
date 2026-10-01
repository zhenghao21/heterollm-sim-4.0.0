"""Extract sm_120 resource records from a CUDA fatbin with cuobjdump."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path

def parse_resource_usage(raw):
 rows=[];arch=None;fn=None
 for line in raw.splitlines():
  m=re.match(r'\s*arch = (\S+)',line)
  if m:arch=m.group(1)
  m=re.match(r' Function (.+):\s*$',line)
  if m:fn=m.group(1)
  m=re.match(r'\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',line)
  if m and fn and arch:
   rows.append({'arch':arch,'function':fn,'registers_per_thread':int(m[1]),'stack_bytes':int(m[2]),'shared_bytes':int(m[3]),'local_bytes':int(m[4])})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('binary',type=Path);ap.add_argument('--cuobjdump',default='cuobjdump.exe');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 raw=subprocess.check_output([a.cuobjdump,'--dump-resource-usage',str(a.binary)],text=True,encoding='utf-8',errors='replace')
 rows=parse_resource_usage(raw)
 data={'schema':'cuda-resource-usage/v1','binary':str(a.binary.resolve()),'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'tool':a.cuobjdump,'records':rows,'notes':['Resource usage is from the exact fatbin, not a hardware specification.','Function-template mangling and c_ncols/flags must be bound to a runtime launch before using a record in a kernel descriptor.','No achieved throughput or bandwidth is inferred from resource usage.']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'records':len(rows),'out':str(a.out)},ensure_ascii=False))
if __name__=='__main__':main()

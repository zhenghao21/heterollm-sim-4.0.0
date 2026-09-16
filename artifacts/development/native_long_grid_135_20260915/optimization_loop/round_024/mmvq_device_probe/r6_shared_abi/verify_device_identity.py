"""Static extraction only; never loads/executes a CUDA module."""
from pathlib import Path
import subprocess, json, hashlib
HERE=Path(__file__).resolve().parent
EX=HERE/'extract_all'
EX.mkdir(exist_ok=True)
p=subprocess.run([r'E:\cuda\bin\cuobjdump.exe','--extract-elf','all',str(HERE/'mmvq-source-slice-link-probe.exe')],cwd=EX,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(HERE/'extract.log').write_bytes(p.stdout)
assert p.returncode == 0
previous=json.loads((HERE.parent/'r5_device_code_compare/device_code_comparison.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
result={'schema':'heterollm.mmvq-shared-abi-device-identity/v1','gpu_execution_performed':False,'timed_runs':0,'pairs':{}}
for name, entry in previous['pairs'].items():
    target=Path(entry['target_cubin']['path'])
    wrapper=EX/Path(entry['wrapper_cubin']['path']).name
    assert sha(target)==entry['target_cubin']['sha256']
    result['pairs'][name]={'target_path':str(target),'target_sha256':sha(target),'wrapper_path':str(wrapper),'wrapper_sha256':sha(wrapper),'bytes':wrapper.stat().st_size,'bytes_equal':wrapper.read_bytes()==target.read_bytes()}
assert all(entry['bytes_equal'] for entry in result['pairs'].values())
(HERE/'device_identity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print('Both device-code segments remain byte-identical; runtime qualification pending.')

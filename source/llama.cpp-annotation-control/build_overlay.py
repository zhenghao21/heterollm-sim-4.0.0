"""Build the annotation overlay without writing to the evidence baseline tree."""
from __future__ import annotations
import ctypes,datetime,hashlib,json,os,pathlib,re,shutil,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent/'llama.cpp-semantic'
OLD=BASE/'build-semantic-direct'
BUILD=ROOT/'build-annotation-control'
EVIDENCE=ROOT/'evidence'
MANIFEST=json.loads((EVIDENCE/'source_manifest.json').read_text(encoding='utf-8'))

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def win_split(command):
    shell=ctypes.windll.shell32
    shell.CommandLineToArgvW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype=ctypes.POINTER(ctypes.c_wchar_p)
    n=ctypes.c_int(); a=shell.CommandLineToArgvW(command,ctypes.byref(n))
    try:return [a[i] for i in range(n.value)]
    finally:ctypes.windll.kernel32.LocalFree(a)

def old_unchanged():
    errors=[]
    for p,sha in MANIFEST['original_runtime_sha256'].items():
        if digest(p)!=sha:errors.append(p)
    for rel,entry in MANIFEST['modified_translation_units'].items():
        if digest(BASE/rel)!=entry['before_sha256']:errors.append(str(BASE/rel))
    if errors:raise RuntimeError('Baseline changed: '+repr(errors))
    return True

old_unchanged()
BUILD.mkdir(exist_ok=True)
(BUILD/'bin').mkdir(exist_ok=True)
for p in (OLD/'bin').iterdir():
    if p.is_file():shutil.copy2(p,BUILD/'bin'/p.name)
commands=json.loads((OLD/'compile_commands.json').read_text(encoding='utf-8'))
vs=pathlib.Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools')
vcvars=vs/'VC/Auxiliary/Build/vcvars64.bat'
env_script=EVIDENCE/'capture_build_environment.cmd'
env_script.write_text('@echo off\ncall "'+str(vcvars)+'" >nul\nset\n',encoding='utf-8')
raw=subprocess.check_output(['cmd.exe','/d','/c',str(env_script)],text=True,encoding='utf-8',errors='replace')
env=dict(os.environ)
for line in raw.splitlines():
    k,sep,v=line.partition('=')
    if sep and k:env[k]=v
compiler=vs/'VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64'
cmake=pathlib.Path('C:/Users/A/AppData/Roaming/Python/Python312/site-packages/cmake/data/bin/cmake.exe')
rc=pathlib.Path('C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64/rc.exe')
mt=rc.with_name('mt.exe')
receipt={'schema':'llama_annotation_control_build_v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(OLD),'build':str(BUILD),'source_manifest_sha256':digest(EVIDENCE/'source_manifest.json'),'old_runtime_preverified':True,'compile_commands_sha256':digest(OLD/'compile_commands.json'),'build_ninja_sha256':digest(OLD/'build.ninja'),'steps':[],'input_sha256':{},'output_sha256':{},'status':'running'}
log=(EVIDENCE/'build.log').open('a',encoding='utf-8')
def run(label,args):
    step={'label':label,'argv':args,'cwd':str(BUILD),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    t=time.monotonic();print(label,flush=True);log.write('\n'+label+'\n'+subprocess.list2cmdline(args)+'\n');log.flush()
    p=subprocess.run(args,cwd=BUILD,env=env,stdout=log,stderr=subprocess.STDOUT)
    step['seconds']=time.monotonic()-t;step['returncode']=p.returncode;receipt['steps'].append(step)
    (EVIDENCE/'build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    if p.returncode:raise RuntimeError(label+' failed; see build.log')

# All relative compiler output flags now resolve below BUILD. Absolute inputs stay read-only.
replacements={}
for rel in MANIFEST['modified_translation_units']:
    entry=next(x for x in commands if pathlib.Path(x['file'])==BASE/rel)
    obj=pathlib.Path(entry['output']).relative_to(OLD); target=BUILD/obj;target.parent.mkdir(parents=True,exist_ok=True)
    args=win_split(entry['command'])
    src=str(BASE/rel)
    args=[str(ROOT/rel) if a.replace('\\','/').lower()==src.replace('\\','/').lower() else a for a in args]
    args.insert(1,'-I'+str(ROOT/'ggml/include'))
    args.insert(2,'-I'+str((BASE/rel).parent))
    run('compile '+rel,args)
    replacements[str(obj).replace('\\','/')]=target
    receipt['input_sha256'][str(ROOT/rel)]=digest(ROOT/rel)
    receipt['output_sha256'][str(target)]=digest(target)
receipt['input_sha256'][str(ROOT/'ggml/include/llama-trace-annotations.h')]=digest(ROOT/'ggml/include/llama-trace-annotations.h')

lines=(OLD/'build.ninja').read_text(encoding='utf-8').splitlines()
def stanza(target):
    wanted=target.replace('\\','/')
    for i,line in enumerate(lines):
        if line.startswith('build ') and wanted in line.split(': ')[0].replace('\\','/').split():
            head,deps=line.split(': ',1);rule,rest=deps.split(' ',1)
            objects=rest.split(' |')[0].split()
            fields={}
            for l in lines[i+1:]:
                if not l.startswith('  '):break
                k,sep,v=l.strip().partition(' = ')
                if sep:fields[k]=v
            return objects,fields
    raise KeyError(target)
def resolve_input(s):
    normalized=s.replace('\\','/')
    p=replacements.get(normalized)
    if p is None:
        raw=pathlib.Path(s)
        p=raw if raw.is_absolute() else OLD/raw
    if not p.is_file():raise FileNotFoundError(p)
    receipt['input_sha256'][str(p)]=digest(p)
    return str(p)

def link(target,mode):
    objects,f=stanza(target)
    objs=[resolve_input(x) for x in objects]
    out=BUILD/pathlib.Path(f['TARGET_FILE']);out.parent.mkdir(parents=True,exist_ok=True)
    if mode=='static':
        args=[str(compiler/'lib.exe'),'/nologo','/machine:x64','/out:'+str(out),*objs]
    else:
        implib=BUILD/pathlib.Path(f['TARGET_IMPLIB']);implib.parent.mkdir(parents=True,exist_ok=True)
        intdir=BUILD/pathlib.Path(f['OBJECT_DIR']);intdir.mkdir(parents=True,exist_ok=True)
        libs=[]
        for val in win_split(f['LINK_LIBRARIES']):
            libs.append(resolve_input(val) if ('\\' in val or '/' in val) else val)
        flags=win_split(f['LINK_FLAGS'])
        flags=[('/DEF:'+resolve_input(x[5:])) if x.startswith('/DEF:') else x for x in flags]
        version='0.23' if 'ggml-cuda' in target else '0.4' if target=='bin/llama.dll' else '0.0'
        rsp=EVIDENCE/(out.stem+'.link.rsp')
        entries=[*objs,*libs,*flags,'/LIBPATH:E:/cuda/lib/x64','/out:'+str(out),'/implib:'+str(implib),'/pdb:'+str(out.with_suffix('.pdb')),'/version:'+version]
        if mode=='dll':entries.append('/dll')
        rsp.write_text('\n'.join(subprocess.list2cmdline([x]) for x in entries),encoding='utf-8')
        args=[str(cmake),'-E','vs_link_dll' if mode=='dll' else 'vs_link_exe','--intdir='+str(intdir),'--rc='+str(rc),'--mt='+str(mt),'--manifests','--',str(compiler/'link.exe'),'/nologo','@'+str(rsp)]
        receipt['input_sha256'][str(rsp)]=digest(rsp)
    run('link '+target,args)
    replacements[f['TARGET_FILE'].replace('\\','/')]=out
    receipt['output_sha256'][str(out)]=digest(out)
    if f.get('TARGET_IMPLIB'):
        implib=BUILD/pathlib.Path(f['TARGET_IMPLIB'])
        if implib.is_file():
            replacements[f['TARGET_IMPLIB'].replace('\\','/')]=implib
            receipt['output_sha256'][str(implib)]=digest(implib)

link('bin/ggml-cuda.dll','dll')
link('bin/llama.dll','dll')
link('tools/server/server-context.lib','static')
link('bin/llama-server-impl.dll','dll')
link('bin/llama-server.exe','exe')
receipt['old_runtime_postverified']=old_unchanged()
receipt['status']='complete';receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt['output_sha256'].update({str(p):digest(p) for p in (BUILD/'bin').iterdir() if p.is_file()})
(EVIDENCE/'build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
log.close();print('COMPLETE '+str(BUILD/'bin/llama-server.exe'),flush=True)

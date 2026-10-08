/** Submit prepared on/off scenarios using real frontend import, validate and run.
 * Run only after native measurements release CPU/GPU and all prepared cases exist.
 * Example: node tools/run_gguf_frontend_matrix.cjs --phase off --ports 8794,8795,8796 --output-dir docs/my-validation
 * Add --resume after restarting this observer to read existing jobs without resubmitting.
 */
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..');
function arg(name,otherwise){const i=process.argv.indexOf(name);if(i<0)return otherwise;const value=process.argv[i+1];assert.ok(typeof value==='string'&&value.trim()&&!value.startsWith('--'),`${name} requires a value`);assert.equal(process.argv.lastIndexOf(name),i,`${name} may only be supplied once`);return value;}
const OUT=path.resolve(arg('--output-dir',path.join(ROOT,'docs/gguf_preset_native_validation_2026-10-08')));
assert.ok(fs.existsSync(OUT)&&fs.statSync(OUT).isDirectory(),'--output-dir must be an existing prepared validation directory');
const phase=arg('--phase',null);assert.ok(['off','on','all'].includes(phase),'--phase off|on|all is required');
const ports=arg('--ports','8794,8795,8796').split(',').map(Number);assert.ok(ports.length&&ports.every(p=>Number.isInteger(p)&&p>=8794&&p<=8813),'Only dedicated validation service ports 8794..8813 are allowed');assert.equal(new Set(ports).size,ports.length);
const resume=process.argv.includes('--resume');
const submitOnly=process.argv.includes('--submit-only');
assert.ok(!(resume&&submitOnly),'--submit-only cannot recover existing jobs; use the GET-only recovery tool');
const cases=JSON.parse(fs.readFileSync(path.join(OUT,'cases.json'),'utf8')).cases;
const selected=arg('--cases',null);const requested=selected?cases.filter(c=>selected.split(',').includes(c.case_id)):cases;
assert.ok(requested.length);const modes=phase==='all'?['off','on']:[phase];
const manifestPath=path.join(OUT,'ui_runs.json');const old=fs.existsSync(manifestPath)?JSON.parse(fs.readFileSync(manifestPath,'utf8')):null;
const manifest=old||{schema:'heterollm.frontend-gguf-validation-runs/v1',methodology:'Real frontend file import, validation, run estimate and background run buttons. Captured actual /api/run-jobs POST body and final UI job snapshot. One job per independent backend service.',runs:[]};
if(submitOnly&&!old)manifest.methodology='Real frontend file import, validation, run estimate and background run buttons. Actual POST body and job creation response are captured. Browser pages close after submission; final reports require serial read-only API collection.';
const transientFileErrors=new Set(['EBUSY','EPERM','EACCES','UNKNOWN']);
const retryWait=new Int32Array(new SharedArrayBuffer(4));let outputSequence=0;
function retryFileOperation(operation){
 for(let attempt=0;;attempt++)try{return operation();}catch(error){
  if(!transientFileErrors.has(error.code)||attempt>=5)throw error;
  // Keep persistence synchronous: no second manifest writer may overtake this one.
  Atomics.wait(retryWait,0,0,Math.min(200,25*2**attempt));
 }
}
function writeJsonAtomic(file,value){
 const temporary=`${file}.${process.pid}.${++outputSequence}.tmp`;const data=JSON.stringify(value,null,2)+'\n';
 try{
  retryFileOperation(()=>fs.writeFileSync(temporary,data));
  retryFileOperation(()=>fs.renameSync(temporary,file));
 }finally{
  try{if(fs.existsSync(temporary))retryFileOperation(()=>fs.unlinkSync(temporary));}catch(error){console.error(JSON.stringify({status:'temporary_file_cleanup_failed',path:temporary,error:error.message}));}
 }
}
function save(){writeJsonAtomic(manifestPath,manifest);}
function output(name,value){writeJsonAtomic(path.join(OUT,name),value);}
function comparable(model){const x=structuredClone(model);delete x.metadata.ui;delete x.metadata.artifact_id;delete x.graph.attributes.ui;x.graph.source_operators??=[];x.graph.sub_operators??=[];x.graph.tensors.sort((a,b)=>a.tensor_id.localeCompare(b.tensor_id));return x;}
function preparedPath(item,mode){return path.join(OUT,`scenario_${item.case_id}_graph_${mode}.json`);}
for(const mode of modes)for(const item of requested){assert.ok(fs.existsSync(preparedPath(item,mode)),`Missing prepared scenario: ${preparedPath(item,mode)}`);const prepared=JSON.parse(fs.readFileSync(preparedPath(item,mode),'utf8'));const base=JSON.parse(fs.readFileSync(path.join(OUT,'base',`scenario_${item.case_id}_512_128.json`),'utf8'));assert.deepStrictEqual(comparable(prepared.model),comparable(base.model),`Prepared ${item.case_id}/${mode} must preserve the selected preset model`);}
const activeStatuses=new Set(['queued','running']);
function isActive(record){return activeStatuses.has(record.status)||(record.job_id&&activeStatuses.has(record.simulation_status)&&record.status!=='completed');}
function recordPort(record){const url=new URL(record.url);assert.equal(url.hostname,'127.0.0.1','Recovery must use the original local validation service');assert.equal(url.protocol,'http:');assert.equal(url.pathname,'/');const port=Number(url.port);assert.ok(ports.includes(port),`Include original port ${port} to recover ${record.case_id}/${record.mode}`);return port;}
// Build every queue before opening a browser. Any prior attempt requires an explicit
// recovery decision; never silently issue a second POST for the same case/mode.
function phasePlan(mode){
 const recoveries=new Map();const pending=[];const skipped=[];
 for(const item of requested){
  const prior=manifest.runs.filter(r=>r.case_id===item.case_id&&r.mode===mode);
  const completed=prior.find(r=>r.status==='completed');
  if(completed){assert.ok(!prior.some(isActive),`Completed case also has an active job: ${item.case_id}/${mode}`);skipped.push(item.case_id);continue;}
  if(!prior.length){pending.push(item);continue;}
  assert.ok(resume,`Prior attempt for ${item.case_id}/${mode}; use --resume for read-only recovery`);
  assert.equal(prior.length,1,`Ambiguous multiple prior attempts for ${item.case_id}/${mode}; inspect manually`);
  const record=prior[0];assert.ok(record.job_id,`Prior attempt ${item.case_id}/${mode} has no recorded job_id; inspect manually, no automatic resubmission`);
  const port=recordPort(record);assert.ok(!recoveries.has(port),`Multiple recorded jobs assigned to port ${port}`);
  assert.ok(record.submission_path&&fs.existsSync(record.submission_path),`Missing original submission for ${item.case_id}/${mode}`);
  assert.ok(record.job_created_path&&fs.existsSync(record.job_created_path),`Missing original creation response for ${item.case_id}/${mode}`);
  const created=JSON.parse(fs.readFileSync(record.job_created_path,'utf8'));assert.equal(created.job_id,record.job_id,'Recorded job identity differs from original creation response');
  const submitted=JSON.parse(fs.readFileSync(record.submission_path,'utf8'));const prepared=JSON.parse(fs.readFileSync(preparedPath(item,mode),'utf8'));assert.deepStrictEqual({...submitted.scenario,model:comparable(submitted.scenario.model)},{...prepared,model:comparable(prepared.model)},'Prepared scenario changed since original frontend submission');
  recoveries.set(port,record);
 }
 return {mode,recoveries,pending,skipped};
}
const plans=modes.map(phasePlan);
const recoveries=new Map();
for(const plan of plans)for(const [port,record] of plan.recoveries){assert.ok(!recoveries.has(port),`Multiple recorded jobs assigned to port ${port} across modes; inspect before recovery`);recoveries.set(port,record);}
// Off jobs enter the shared queue first. On jobs use reversed requested order;
// this changes observer scheduling only, never the prepared simulation inputs.
const pending=plans.flatMap(plan=>(plan.mode==='on'?[...plan.pending].reverse():plan.pending).map(item=>({item,mode:plan.mode})));
if(submitOnly)assert.ok(recoveries.size===0&&pending.length<=ports.length,'--submit-only requires a separate free service port for every pending job');
for(const record of manifest.runs.filter(isActive)){
 const port=Number(new URL(record.url).port);if(!ports.includes(port))continue;
 assert.equal(recoveries.get(port),record,`Port ${port} has an active job outside the selected cases/modes; include it for recovery`);
}
if(process.argv.includes('--check-only')){console.log(JSON.stringify({status:'inputs_checked_no_simulation',cases:requested.length,modes,ports,resume,recoveries:[...recoveries].map(([port,r])=>({port,case_id:r.case_id,mode:r.mode,job_id:r.job_id})),pending:pending.map(({item,mode})=>({case_id:item.case_id,mode})),skipped:plans.flatMap(p=>p.skipped.map(case_id=>({case_id,mode:p.mode}))),scheduling:'Shared queue; existing jobs stay on original ports; new off jobs submitted before new on jobs; no completion barrier; on order reverses requested order without changing inputs.'}));process.exit(0);}
let halt=false;
async function recoverOne(record,port){
 const stem=`ui_${record.case_id}_graph_${record.mode}`;const observerBegin=Date.now();const originalBegin=Date.parse(record.started_at);let previous='';let lastLog=0;let readFailures=0;
 assert.ok(Number.isFinite(originalBegin),'Recorded start time is invalid');
 record.observer_recovery='read-only API after observer restart';record.recovery_started_at=new Date().toISOString();record.observer_status='polling';delete record.observer_error;
 manifest.recovery_methodology='Previously submitted frontend jobs are recovered only through GET /api/run-jobs/{job_id} on their original service. Original submission and job_created files are retained. Recovered final snapshots come from the read-only API, not a continued frontend page; report rendering after recovery is not revalidated.';save();
 console.log(JSON.stringify({case_id:record.case_id,mode:record.mode,port,status:'recovering_existing_job',job_id:record.job_id}));
 try{
  for(;;){
   let current;
   try{
    const response=await fetch(new URL(`api/run-jobs/${encodeURIComponent(record.job_id)}`,record.url),{method:'GET',signal:AbortSignal.timeout(30000)});
    if(!response.ok){const error=new Error(`Read-only recovery HTTP ${response.status}; existing job will not be resubmitted`);error.permanent=response.status>=400&&response.status<500;throw error;}
    current=await response.json();assert.equal(current.job_id,record.job_id,'Recovery returned a different job');assert.ok(['queued','running','completed','failed','cancelled'].includes(current.status),'Recovery returned an unknown job status');readFailures=0;
   }catch(error){
    readFailures++;record.recovery_read_failures=readFailures;record.observer_error=error.message;save();
    if(error.permanent||readFailures>=5)throw error;
    await new Promise(resolve=>setTimeout(resolve,5000));continue;
   }
   const elapsed=(Date.now()-originalBegin)/1000;const summary={status:current.status,job_id:current.job_id,error:current.error,progress:current.progress,report_ready:!!current.report,observer_recovery:record.observer_recovery};const signature=JSON.stringify(summary);
   if(signature!==previous){fs.appendFileSync(record.progress_path,JSON.stringify({elapsed_seconds:elapsed,...summary})+'\n');previous=signature;record.status=current.status;record.wall_seconds=elapsed;record.last_progress=current.progress;delete record.observer_error;save();}
   if(Date.now()-lastLog>=30000){console.log(JSON.stringify({case_id:record.case_id,mode:record.mode,port,status:current.status,wall_seconds:elapsed,progress:current.progress,observer_recovery:record.observer_recovery}));lastLog=Date.now();}
   if(['completed','failed','cancelled'].includes(current.status)){
    output(`${stem}_result.json`,current);record.result_path=path.join(OUT,`${stem}_result.json`);record.finished_at=current.finished_at||new Date().toISOString();record.wall_seconds=elapsed;record.status=current.status;record.observer_status='finished';record.recovered_report_rendering_checked=false;delete record.simulation_status;save();
    if(current.status!=='completed')throw new Error('Simulation '+current.status+': '+JSON.stringify(current.error));
    assert.ok(current.report,'Completed recovery snapshot has no report');
    console.log(JSON.stringify({case_id:record.case_id,mode:record.mode,port,status:'completed',job_id:record.job_id,wall_seconds:elapsed,observer_recovery:record.observer_recovery}));return;
   }
   if(Date.now()-observerBegin>10800000)throw new Error('Three-hour recovery observer limit reached; preserve active job for another explicit --resume');
   await new Promise(resolve=>setTimeout(resolve,5000));
  }
 }catch(error){halt=true;record.observer_status='failed';record.observer_error=error.message.split('\n')[0];record.observer_finished_at=new Date().toISOString();save();console.error(JSON.stringify({case_id:record.case_id,mode:record.mode,port,status:'observer_failed',simulation_status:record.status,error:record.observer_error,job_id:record.job_id,automatic_resubmission:false}));}
}
async function runOne(browser,item,mode,port){
 const file=preparedPath(item,mode);const prepared=JSON.parse(fs.readFileSync(file,'utf8'));const stem=`ui_${item.case_id}_graph_${mode}`;
 const existing=manifest.runs.find(r=>r.case_id===item.case_id&&r.mode===mode&&r.status==='completed');
 if(existing){console.log(JSON.stringify({case_id:item.case_id,mode,status:'already_completed',job_id:existing.job_id}));return;}
 const record={case_id:item.case_id,preset_id:item.preset_id,mode,url:`http://127.0.0.1:${port}/`,scenario_path:file,status:'preparing',started_at:new Date().toISOString(),page_errors:[],console_errors:[],progress_path:path.join(OUT,`${stem}_progress.jsonl`)};manifest.runs.push(record);save();
 const page=await browser.newPage({acceptDownloads:true});page.setDefaultTimeout(180000);const begin=Date.now();let lastLog=0;
 page.on('pageerror',error=>{record.page_errors.push(error.message);try{save();}catch(writeError){halt=true;record.observer_error=writeError.message;console.error(JSON.stringify({case_id:item.case_id,mode,port,status:'observer_persistence_failed',error:writeError.message,job_id:record.job_id}));}});page.on('console',message=>{if(message.type()==='error')record.console_errors.push(message.text());});page.on('dialog',dialog=>dialog.accept());
 try{
  await page.goto(record.url);await page.waitForFunction(()=>state.scenario!==null);
  const chooserPromise=page.waitForEvent('filechooser');await page.locator('#importButton').click();const chooser=await chooserPromise;
  const normalizedPromise=page.waitForResponse(r=>r.url().endsWith('/api/normalize')&&r.request().method()==='POST');await chooser.setFiles(file);const normalized=await normalizedPromise;if(!normalized.ok())throw new Error(`Import rejected HTTP ${normalized.status()}: ${await normalized.text()}`);await page.waitForFunction(name=>state.scenario.name===name&&!state.busy,prepared.name);
  const validationPromise=page.waitForResponse(r=>r.url().endsWith('/api/validate')&&r.request().method()==='POST');await page.locator('#validateButton').click();record.validation=await(await validationPromise).json();if(!record.validation.valid)throw new Error('Frontend validation failed');save();
  await page.locator('#runButton').click();await page.waitForFunction(()=>!state.busy&&state.runEstimate!==null);
  record.estimate=await page.evaluate(()=>state.runEstimate);save();
  const createdPromise=page.waitForResponse(r=>r.url().endsWith('/api/run-jobs')&&r.request().method()==='POST');await page.locator('#startRunJobButton').click();const createdResponse=await createdPromise;
  const submitted=createdResponse.request().postDataJSON();output(`${stem}_submission.json`,submitted);record.submission_path=path.join(OUT,`${stem}_submission.json`);
  assert.deepStrictEqual(comparable(submitted.scenario.model),comparable(prepared.model),'Actual submitted model differs from selected preset');
  for(const key of ['context','batch','ubatch','threads','threads_batch','flash_attn','kv_type_k','kv_type_v','parallel','gpu_layers'])assert.equal(submitted.scenario.profiles.llama_cpp[key],prepared.profiles.llama_cpp[key],`Actual ${key} differs`);
  record.model_semantics_preserved=true;record.retention_policy=submitted.retention_policy;
  const created=await createdResponse.json();output(`${stem}_job_created.json`,created);record.job_created_path=path.join(OUT,`${stem}_job_created.json`);if(!createdResponse.ok())throw new Error('Job creation rejected: '+JSON.stringify(created));record.job_id=created.job_id;record.status=created.status;save();
  console.log(JSON.stringify({case_id:item.case_id,mode,port,status:record.status,job_id:record.job_id}));
  if(submitOnly){
   assert.equal(record.page_errors.length,0,'Frontend submission raised a page exception');
   record.observer_status='submitted_for_read_only_collection';
   record.recovered_report_rendering_checked=false;
   manifest.recovery_methodology='Scenarios are imported, validated and submitted through the actual frontend. Pages close after job creation; final reports are collected serially through the read-only API. This does not claim sustained frontend result rendering.';
   save();return;
  }
  let previous='';
  for(;;){
   const current=await page.evaluate(()=>({status:state.runJob?.status,job_id:state.runJob?.job_id,error:state.runJob?.error,progress:state.runJob?.progress,report_ready:!!state.report,report_stale:state.reportStale,poll_failures:state.runJobPollFailures}));
   const signature=JSON.stringify(current);const elapsed=(Date.now()-begin)/1000;
   if(signature!==previous){fs.appendFileSync(record.progress_path,JSON.stringify({elapsed_seconds:elapsed,...current})+'\n');previous=signature;record.status=current.status;record.wall_seconds=elapsed;record.last_progress=current.progress;save();}
   if(Date.now()-lastLog>=30000){console.log(JSON.stringify({case_id:item.case_id,mode,port,status:current.status,wall_seconds:elapsed,progress:current.progress}));lastLog=Date.now();}
   if(['completed','failed','cancelled'].includes(current.status)){
    const result=await page.evaluate(()=>state.runJob);output(`${stem}_result.json`,result);record.result_path=path.join(OUT,`${stem}_result.json`);record.finished_at=new Date().toISOString();record.wall_seconds=elapsed;record.status=current.status;record.report_stale=current.report_stale;save();
    if(current.status!=='completed')throw new Error('Simulation '+current.status+': '+JSON.stringify(current.error));
    await page.waitForFunction(()=>!!state.report);assert.equal(current.report_stale,false,'Frontend report unexpectedly stale');assert.equal(record.page_errors.length,0,'Frontend page exception occurred');
    console.log(JSON.stringify({case_id:item.case_id,mode,port,status:'completed',job_id:record.job_id,wall_seconds:record.wall_seconds}));break;
   }
   if(current.poll_failures>=5)throw new Error('Frontend stopped polling job; preserve job_id for recovery');
   if(elapsed>10800)throw new Error('Three-hour observer limit reached; preserve active job for recovery');
   await page.waitForTimeout(5000);
  }
 }catch(error){halt=true;record.observer_status='failed';record.observer_error=error.message.split('\n')[0];record.wall_seconds=(Date.now()-begin)/1000;record.observer_finished_at=new Date().toISOString();if(!record.job_id)record.status='failed';console.error(JSON.stringify({case_id:item.case_id,mode,port,status:'observer_failed',simulation_status:record.status,error:record.observer_error,job_id:record.job_id,automatic_resubmission:false}));save();}
 finally{await page.close();}
}
function offJobsSubmitted(){return !modes.includes('off')||requested.every(item=>manifest.runs.some(r=>r.case_id===item.case_id&&r.mode==='off'&&(r.job_id||r.status==='completed')));}
async function runQueue(browser){
 let index=0;
 manifest.scheduling_methodology='One shared queue across Graph modes. Existing jobs recover on their original ports. New off jobs are all submitted before new on jobs start, but off completion is not a barrier. On jobs use reversed requested order; prepared inputs and simulation costs are unchanged.';save();
 for(const plan of plans)for(const case_id of plan.skipped)console.log(JSON.stringify({case_id,mode:plan.mode,status:'already_completed'}));
 const settled=await Promise.allSettled(ports.map(async port=>{
  try{
   const recovery=recoveries.get(port);if(recovery)await recoverOne(recovery,port);
   while(!halt&&index<pending.length){
    // A free worker may reach on while another worker is still using the UI to
    // submit its off job. Wait only for those POST results, not their simulation.
    if(pending[index].mode==='on'&&!offJobsSubmitted()){await new Promise(resolve=>setTimeout(resolve,250));continue;}
    const {item,mode}=pending[index++];await runOne(browser,item,mode,port);
    if(submitOnly)break;
   }
  }catch(error){halt=true;console.error(JSON.stringify({port,status:'observer_worker_failed',error:error.message.split('\n')[0],automatic_resubmission:false}));throw error;}
 }));
 if(settled.some(result=>result.status==='rejected'))process.exitCode=1;
}
(async()=>{const {chromium}=require('C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');const browser=await chromium.launch({headless:true,channel:'msedge'});try{await runQueue(browser);}finally{await browser.close();manifest.updated_at=new Date().toISOString();save();}if(halt)process.exitCode=1;})().catch(error=>{console.error(error.message.split('\n')[0]);process.exitCode=1});

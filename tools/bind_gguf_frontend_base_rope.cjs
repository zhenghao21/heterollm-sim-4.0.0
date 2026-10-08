/** Bind the fixed source-derived RoPE execution contract through frontend import.
 * Existing preset-derived model data remains unchanged; all source fields are
 * explicit and no native timing is read.
 */
const {chromium}=require('C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..');const OUT=path.join(ROOT,'docs/gguf_preset_native_validation_2026-10-08');
const cases=JSON.parse(fs.readFileSync(path.join(OUT,'cases.json'),'utf8')).cases;
const manifestPath=path.join(OUT,'ui_preparation.json');const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
const scratch=path.resolve(ROOT,'../../_scratch/37-gguf-ui-base-import');fs.mkdirSync(scratch,{recursive:true});
function comparable(model){const x=structuredClone(model);delete x.metadata.ui;delete x.metadata.artifact_id;delete x.graph.attributes.ui;x.graph.source_operators??=[];x.graph.sub_operators??=[];x.graph.tensors.sort((a,b)=>a.tensor_id.localeCompare(b.tensor_id));return x;}
function save(){fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');}
(async()=>{const browser=await chromium.launch({headless:true,channel:'msedge'});try{for(const item of cases){
 const outputPath=path.join(OUT,'base',`scenario_${item.case_id}_512_128.json`);const payload=JSON.parse(fs.readFileSync(outputPath,'utf8'));const model=structuredClone(payload.model);
 const architecture=payload.model.metadata.metadata.gguf_architecture_id;assert.ok(['qwen3','qwen35','qwen2','llama'].includes(architecture),`Unqualified RoPE architecture ${architecture}`);
 const contract={schema:'llama.cpp.cuda-rope/v1',strategy:'runtime_sin_cos',position_components:architecture==='qwen35'?4:1,source_revision:'d3146f2b56c2db4711ac8391871c9e529d1946d7',timing_completeness:'partial'};
 payload.workload.metadata.native_rope_source_contract=contract;
 const inputPath=path.join(scratch,`${item.case_id}.rope-configured.json`);fs.writeFileSync(inputPath,JSON.stringify(payload,null,2)+'\n');
 const record=manifest.cases.find(row=>row.case_id===item.case_id);assert.ok(record);const binding={method:'Add explicit source contract to auxiliary JSON, then real frontend import, validate and export buttons',architecture,contract,input_path:inputPath,page_errors:[],status:'in_progress'};record.rope_source_binding=binding;save();
 const page=await browser.newPage({acceptDownloads:true});page.setDefaultTimeout(180000);page.on('pageerror',e=>binding.page_errors.push(e.message));page.on('dialog',d=>d.accept());const started=Date.now();
 try{
  await page.goto('http://127.0.0.1:8794/');await page.waitForFunction(()=>state.scenario!==null);
  const choosePromise=page.waitForEvent('filechooser');await page.locator('#importButton').click();const chooser=await choosePromise;const normPromise=page.waitForResponse(r=>r.url().endsWith('/api/normalize')&&r.request().method()==='POST');await chooser.setFiles(inputPath);const normalized=await normPromise;if(!normalized.ok())throw new Error('RoPE import failed: '+await normalized.text());await page.waitForFunction(name=>state.scenario.name===name&&!state.busy,payload.name);
  const validationPromise=page.waitForResponse(r=>r.url().endsWith('/api/validate')&&r.request().method()==='POST');await page.locator('#validateButton').click();binding.validation=await(await validationPromise).json();assert.equal(binding.validation.valid,true,'RoPE-bound frontend scenario validation failed');
  const downloadPromise=page.waitForEvent('download');await page.locator('#exportButton').click();await(await downloadPromise).saveAs(outputPath);const exported=JSON.parse(fs.readFileSync(outputPath,'utf8'));assert.deepStrictEqual(comparable(exported.model),comparable(model));assert.deepStrictEqual(exported.workload.metadata.native_rope_source_contract,contract);assert.equal(binding.page_errors.length,0);binding.status='completed';binding.model_semantics_preserved=true;binding.wall_seconds=(Date.now()-started)/1000;record.export_bytes=fs.statSync(outputPath).size;save();console.log(JSON.stringify({case_id:item.case_id,status:binding.status,page_errors:binding.page_errors}));
 }catch(error){binding.status='failed';binding.error=error.message.split('\n')[0];save();throw error;}finally{await page.close();}
 }}finally{await browser.close();}})().catch(error=>{console.error(error.message.split('\n')[0]);process.exitCode=1});

"""Host-only safety/ordering tests; never execute the GPU probe binary."""
from pathlib import Path
import hashlib,json,subprocess,unittest
P=Path(__file__).resolve().parent
class ModulePreload(unittest.TestCase):
 def test_fake_host_guard_lifetime_and_rejections(self):
  r=subprocess.run([str(P/'module-guard-host.exe')],capture_output=True,text=True,check=True)
  d=json.loads(r.stdout);self.assertEqual(d['checks'],8);self.assertFalse(d['gpu_access']);self.assertFalse(d['native_libraries_loaded'])
 def test_preload_and_both_identity_checks_precede_cuda(self):
  s=(P/'stream_event_probe.cpp').read_text(encoding='utf-8-sig');body=s[s.index('static int run('):]
  self.assertLess(body.index('verify_files();'),body.index('pinned_modules.preload(frozen_modules);'))
  self.assertLess(body.index('pinned_modules.preload(frozen_modules);'),body.index('verify_loaded_modules();'))
  self.assertLess(body.index('verify_loaded_modules();'),body.index('cudaSetDevice('))
  self.assertLess(body.index('PinnedFrozenModules<FrozenNativeModuleApi>'),body.index('Resources r;'))
  self.assertEqual(body.count('verify_loaded_modules();'),3)
 def test_fixed_absolute_loader_and_live_handle_verification(self):
  s=(P/'stream_event_probe.cpp').read_text(encoding='utf-8-sig')
  self.assertIn('LoadLibraryExA(path, nullptr,',s);self.assertIn('LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS',s)
  self.assertIn('GetModuleFileNameA(handle, actual',s);self.assertIn('file_hash(actual) != expected_hash',s)
  self.assertIn("text[1] != ':'",s);self.assertIn('drive-absolute',s)
 def test_all_four_native_modules_and_local_dependencies_locked(self):
  lock=json.loads((P/'identity_lock.json').read_text());refs=lock['files']
  names={Path(r['path']).name for r in refs if Path(r['path']).suffix.lower()=='.dll'}
  self.assertTrue({'ggml.dll','ggml-base.dll','ggml-cpu.dll','ggml-cuda.dll'}<=names)
  for n in ('protocol.json','source_reference.h','frozen_module_guard.h','stream_event_probe.cpp'):
   p=P/n;r=next(r for r in refs if Path(r['path'])==p)
   self.assertEqual(r['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
 def test_prior_frozen_probe_files_unchanged(self):
  provenance=json.loads((P/'revision_provenance.json').read_text())
  for r in provenance['source_files']:
   self.assertEqual(r['sha256'],hashlib.sha256(Path(r['path']).read_bytes()).hexdigest())
 def test_no_gpu_probe_called_by_host_test(self):
  s=(P/'module_guard_host.cpp').read_text(encoding='utf-8-sig')
  self.assertNotIn('cudaSetDevice',s);self.assertNotIn('cuda_runtime',s);self.assertNotIn('LoadLibrary',s)
 def test_raw_timing_numeric_semantics_unchanged(self):
  old=json.loads((P.parent/'r2/protocol.json').read_text());new=json.loads((P/'protocol.json').read_text())
  for name in ('configs','execution','correctness','quality_gates','failure_policy'):
   self.assertEqual(old[name],new[name])
if __name__=='__main__':unittest.main(verbosity=2)

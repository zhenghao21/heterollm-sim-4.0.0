import tempfile,unittest,json
from pathlib import Path
from bridge import inventory,schema_readiness,write_new,load
class Preparation(unittest.TestCase):
 def test_no_measurement_read_without_root_confirmation(self):
  with self.assertRaisesRegex(ValueError,'root confirmation'):
   inventory(Path('absent'),Path('absent'),Path('absent'),Path('absent'))
 def test_frozen_v2_cannot_silently_feed_v1_resolver(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);c=p/'collector';b=p/'probe';c.mkdir();b.mkdir();source=p/'resolver.py'
   write_new(c/'protocol.json',{'schema':'operator-matrix-collection-protocol/v2','quality_policy':{'every_formal_interval_bracketed_required':True}})
   write_new(b/'protocol.json',{'schema':'synthetic'})
   source.write_text('protocol.get("schema") != "operator-matrix-collection-protocol/v1"')
   r=schema_readiness(c,b,source)
   self.assertFalse(r['can_emit_accepted_profile']);self.assertEqual(len(r['blockers']),2);self.assertFalse(r['measurements_read'])
 def test_output_never_overwrites(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'result.json';write_new(p,{'original':True})
   with self.assertRaises(FileExistsError):write_new(p,{'overwritten':True})
   self.assertEqual(load(p),{'original':True})
if __name__=='__main__':unittest.main(verbosity=2)

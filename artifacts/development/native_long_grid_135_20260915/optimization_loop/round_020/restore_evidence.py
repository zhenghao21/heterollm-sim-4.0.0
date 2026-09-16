"""Restore byte-exact R20 evidence from the compact Git packet; never overwrite."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
index=json.loads((ROOT/'detailed_evidence_index.json').read_text(encoding='utf-8'))
packet=ROOT/'detailed_evidence.tar.gz'
assert hashlib.sha256(packet.read_bytes()).hexdigest()==index['packet_sha256'],'packet identity mismatch'
assert index['members'],'empty evidence index'
expected={r['relative_path']:r for r in index['members']}
assert len(expected)==len(index['members']),'duplicate evidence path'
restored=verified=0
with tarfile.open(packet,'r:gz') as tar:
    entries=tar.getmembers()
    assert len(entries)==len(expected) and {m.name for m in entries}==set(expected),'unexpected archive member'
    for member in entries:
        assert member.isfile(),'regular evidence files only'
        dest=(ROOT/member.name).resolve()
        dest.relative_to(ROOT) # reject absolute/parent/symlink escapes
        item=expected[member.name];content=tar.extractfile(member).read()
        assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256'],'member hash mismatch'
        if dest.exists():
            assert dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'],'refuse to overwrite divergent evidence'
            verified+=1
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(content)
            restored+=1
print(json.dumps({'restored':restored,'already_verified':verified,'members':len(expected)}))
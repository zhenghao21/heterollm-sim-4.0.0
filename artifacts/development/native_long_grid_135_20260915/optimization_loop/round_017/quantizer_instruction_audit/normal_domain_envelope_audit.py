"""Conditional reciprocal-contract arithmetic audit. Diagnostic only, never a gate patch."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from collections import Counter
import hashlib,json,math,struct
P=Path(__file__).resolve().parent
R=P.parent.parent/'round_016/conversion_capture/replay_r1'
M,K=64,1024
u=F(1,2**24)
# Covers both a one-ULP error measured from an exact reciprocal and a one-ULP
# error measured from its correctly rounded value, with RN error included.
rho=3*u+2*u*u
alpha=(1+rho)*(1+u)-1
tau=(1+rho)*(1+u)**2-1
minimum_normal=2.0**-126

def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(b):return struct.unpack('<f',struct.pack('<I',b))[0]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def rational(x):return F.from_float(x)
def power2(e):return F(2**e) if e>=0 else F(1,2**-e)
def normal(x):return math.isfinite(x) and abs(x)>=minimum_normal

def rn32_exact_positive(x):
 if x<=0:raise ValueError('positive rational required')
 e=x.numerator.bit_length()-x.denominator.bit_length()
 if x<power2(e):e-=1
 if e < -126 or e>127:raise ValueError('reference outside normal finite reciprocal scope')
 t=x/power2(e-23);q,r=divmod(t.numerator,t.denominator)
 if 2*r>t.denominator or (2*r==t.denominator and q%2):q+=1
 if q==2**24:q//=2;e+=1
 if e>127:raise ValueError('rounded reciprocal overflow')
 return value(((e+127)<<23)+(q-2**23))

@lru_cache(maxsize=None)
def rcp_candidates(a):
 if not normal(a) or a<=0:raise ValueError('amax/inverse outside positive normal scope')
 exact=1/rational(a);center=bits(rn32_exact_positive(exact))
 lo=exact*(1-rho);hi=exact*(1+rho)
 candidates=[]
 # rho is under 2^-22: this bounded neighborhood also covers binade boundaries.
 for b in range(max(0,center-8),min(0x7f800000,center+9)):
  r=value(b)
  if normal(r) and lo<=rational(r)<=hi:candidates.append(r)
 if not candidates:raise AssertionError('normal reciprocal envelope unexpectedly empty')
 # Every exterior neighbor is already outside the rational interval.
 if center>8:assert rational(value(center-8))<lo
 if center+8<0x7f800000:assert rational(value(center+8))>hi
 return tuple(candidates)

def mul_normal(a,b):
 # Product of two binary32 operands is exact in binary64 before final RN32.
 try:y=f32(a*b)
 except OverflowError:raise ValueError('multiplication overflow outside certified normal domain')
 if y!=0 and not normal(y):raise ValueError('FTZ product case outside certified normal domain')
 return y

def round_away(z):return int(math.copysign(math.floor(abs(z)+0.5),z))
def decode_qd(data,m,b):
 off=((b//4)*M+m)*144
 d=struct.unpack_from('<f',data,off+(b%4)*4)[0]
 q=struct.unpack_from('<32b',data,off+16+(b%4)*32)
 return d,q

def test_host():
 for x in (F(1),F(2),F(1,2),F(127,256),F(1,2**126),F(2**100)):
  assert rational(rn32_exact_positive(x))==x
 assert rn32_exact_positive(F(1)+F(1,2**24))==1.0
 assert bits(rn32_exact_positive(F(1)+F(3,2**24)))==bits(1.0)+2
 for a in (0.25,0.5,1.0,2.0,127.0,256.0,minimum_normal):
  for candidate in rcp_candidates(a):assert abs(rational(candidate)-1/rational(a))<=rho/rational(a)
 assert round_away(63.5)==64 and round_away(value(bits(63.5)-1))==63
 assert round_away(-63.5)==-64 and round_away(-value(bits(63.5)-1))==-63
 # PTX LOP3 truth-table constants ta=f0,tb=cc,tc=aa imply
 # (a & ~b) | (c & b) = b8: preserve 0.5 magnitude and use z sign.
 assert ((0xf0 & (~0xcc & 0xff)) | (0xaa & 0xcc))==0xb8
 for z in (-63.5,63.5,-0.0,0.0):
  half_bits=(0x3f000000 & 0x7fffffff) | (bits(z) & 0x80000000)
  assert abs(value(half_bits))==0.5
 return {'exact_positive_RN32_ties_test':True,'reciprocal_envelope_boundary_test':True,'half_away_discontinuity_test':True,'LOP3_sign_half_truth_table':True,'GPU_executed':False}

def ref(path):
 with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha}

def main():
 tests=test_host()
 xp=R/'replay.0001.json.input_f32.bin';qp=R/'replay.0001.json.q8_1_d4.bin'
 x=struct.unpack('<65536f',xp.read_bytes());raw=qp.read_bytes()
 assert len(raw)==73728
 failures=[];unsupported=[];target=None;ambiguous_code_count=0;ambiguous_blocks=0;joint_count=0;scale_hist=Counter();independent_q_failures=0;independent_scale_failures=0
 for m in range(M):
  for b in range(K//32):
   xs=x[m*K+b*32:m*K+(b+1)*32];d,q=decode_qd(raw,m,b)
   try:
    if any(not math.isfinite(a) or (a!=0 and not normal(a)) for a in xs):raise ValueError('non-finite/subnormal source input')
    a=max(abs(v) for v in xs)
    inv_to_rcp={}
    for r1 in rcp_candidates(a):inv_to_rcp.setdefault(mul_normal(127.0,r1),[]).append(r1)
    paths=[]
    for inv,firsts in sorted(inv_to_rcp.items()):
     codes=tuple(round_away(mul_normal(z,inv)) for z in xs)
     scale_values=rcp_candidates(inv)
     paths.append({'inverse':inv,'inverse_bits':bits(inv),'possible_first_reciprocals':firsts,'codes':codes,'possible_scale_bits':[bits(v) for v in scale_values],'codes_match':codes==q,'scale_matches':bits(d) in [bits(v) for v in scale_values]})
    joint=[path for path in paths if path['codes_match'] and path['scale_matches']]
    allowed=[sorted({path['codes'][i] for path in paths}) for i in range(32)]
    amb=sum(len(c)>1 for c in allowed);ambiguous_code_count+=amb;ambiguous_blocks+=bool(amb)
    independent_q_failures+=sum(q[i] not in allowed[i] for i in range(32))
    independent_scale_failures+=not any(path['scale_matches'] for path in paths)
    if joint:joint_count+=1
    else:failures.append({'row':m,'block32':b,'actual_d4':d,'actual_q':q,'paths':paths})
    cpu_inverse=f32(127.0/a);cpu_scale=f32(1.0/cpu_inverse)
    scale_hist[bits(d)-bits(cpu_scale)]+=1
    if m==11 and b==1:
     target={'row':m,'block32':b,'amax':a,'amax_bits':bits(a),'k33_x':xs[1],'ideal_scaled_value':float(F(127)*rational(xs[1])/rational(a)),'actual_q_k33':q[1],'actual_d4':d,'actual_d4_bits':bits(d),'allowed_k33_codes':allowed[1],'candidate_paths':[dict(path,codes=list(path['codes']),q_k33=path['codes'][1]) for path in paths],'joint_witness_inverse_bits':[path['inverse_bits'] for path in joint],'joint_witnesses_are_not_observed_registers':True}
   except ValueError as e:unsupported.append({'row':m,'block32':b,'reason':str(e)})
 result={'schema':'conditional-normal-domain-reciprocal-envelope/v1','diagnostic_only_no_gate_acceptance':True,'GPU_executed':False,'no_tolerance_or_reference_patch':True,'host_tests':tests,'inputs':[ref(xp),ref(qp)],'contract_status':'CONDITIONAL. PTX rcp.approx.f32 has documented max 1 ULP, but active original DLL has no embedded PTX. MUFU.RCP mnemonic alone does not establish this exact numeric contract for this deployed SASS. No production conversion pass may be inferred.','normal_domain':'Finite binary32 inputs, each nonzero input normal; amax>0; all enumerated reciprocals, inverse and nonzero scaled products finite normal. Zero blocks, subnormals, overflow/underflow, NaN and infinity remain unsupported, not passed.','reciprocal_relative_bound':{'u':float(u),'rho':float(rho),'exact_formula':'rho=3u+2u^2; conservative bound covering RN-reference ULP convention'},'inverse_relative_bound':{'alpha':float(alpha),'formula':'(1+rho)*(1+u)-1'},'scaled_product_relative_bound':{'tau':float(tau),'formula':'(1+rho)*(1+u)^2-1','absolute_upper_bound_if_abs_x_le_amax':float(127*tau)},'scale_relative_interval':{'lower_ratio':float((1-rho)/(1+alpha)),'upper_ratio':float((1+rho)/(1-alpha)),'formula':'(1-rho)/(1+alpha) <= d/(amax/127) <= (1+rho)/(1-alpha)'},'blocks_examined':2048,'normal_supported_blocks':2048-len(unsupported),'unsupported_blocks':unsupported,'blocks_with_joint_32_codes_and_scale_witness':joint_count,'joint_witness_failure_count':len(failures),'joint_witness_failures':failures[:32],'independent_code_outside_envelope_count':independent_q_failures,'independent_scale_outside_envelope_count':independent_scale_failures,'codes_with_multiple_permitted_values':ambiguous_code_count,'blocks_with_ambiguous_codes':ambiguous_blocks,'observed_scale_minus_CPU_scale_bit_steps_histogram':dict(sorted(scale_hist.items())),'target_block':target,'portable_exact_reference_obstruction':'Same allowed normal-domain reciprocal contract permits q=63 and q=64 at ideal scaled value 63.5. A CPU point oracle cannot choose the hardware result from source arithmetic/error bounds alone. This is input-generic half-integer discontinuity, never an index-specific exception.'}
 with (P/'normal_domain_envelope_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('target_block','joint_witness_failures','inputs')},ensure_ascii=False))
 print(json.dumps({'target':{k:v for k,v in target.items() if k!='candidate_paths'}},ensure_ascii=False))
if __name__=='__main__':main()

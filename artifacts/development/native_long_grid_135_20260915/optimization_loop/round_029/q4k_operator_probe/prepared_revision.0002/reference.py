"""Independent packed-Q4_K/Q8_1 CPU fixture and interval, no GGML imports."""
from pathlib import Path
import hashlib,json,math,struct
K=N=2048
Q4_BYTES=144
Q8_BYTES=36
ROUNDING_CAP=2*K+64
HERE=Path(__file__).resolve().parent


def need(ok,message):
    if not ok:raise ValueError(message)


def pack_scales(scales,mins):
    need(len(scales)==len(mins)==8 and all(0<=x<64 for x in scales+mins),'six-bit scale/min')
    out=bytearray(12)
    for j in range(4):
        out[j]=scales[j]|((scales[j+4]>>4)<<6)
        out[j+4]=mins[j]|((mins[j+4]>>4)<<6)
        out[j+8]=(scales[j+4]&15)|((mins[j+4]&15)<<4)
    return bytes(out)


def decode_scales(data,j):
    if j<4:return data[j]&63,data[j+4]&63
    return (data[j+4]&15)|((data[j-4]>>6)<<4),(data[j+4]>>4)|((data[j]>>6)<<4)


def make_weights(n=N):
    result=bytearray()
    for row in range(n):
        for block in range(K//256):
            scales=[1+((row*11+block*7+j*13)%63) for j in range(8)]
            mins=[1+((row*17+block*19+j*23)%63) for j in range(8)]
            d=2.0**(-10+row%2);dm=2.0**(-11+block%2)
            out=bytearray(struct.pack('<ee',d,dm)+pack_scales(scales,mins)+bytes(128))
            for group in range(4):
                for j in range(32):
                    lo=(row*3+block*5+group*7+j*11)%16
                    hi=(row*7+block*11+group*3+j*5+1)%16
                    out[16+group*32+j]=lo|(hi<<4)
            result.extend(out)
    return bytes(result)


def make_input():
    result=[]
    for b in range(K//32):
        d=2.0**(-10+b%4)
        for j in range(15):
            q=1+(b*17+j*29)%120
            result.extend((d*q,-d*q))
        result.extend((d*127,-d*(64+b%63)))
    return struct.pack('<%df'%K,*result)


def expected_q8(input_bytes):
    x=struct.unpack('<%df'%K,input_bytes);out=bytearray()
    for b in range(K//32):
        values=x[b*32:(b+1)*32];d=max(map(abs,values))/127;total=sum(values)
        need(total!=0 and d>0,'nonzero sum/min correction required')
        q=[int(round(v/d)) for v in values]
        need(all(v/d==z and -127<=z<=127 and z!=0 for v,z in zip(values,q)),'exact dyadic quantization required')
        half=struct.pack('<ee',d,total)
        need(struct.unpack('<ee',half)==(d,total),'d and original-input sum must be exact binary16')
        need(d*sum(q)==total,'Q8 original sum differs from quantized sum')
        out.extend(half+struct.pack('<32b',*q))
    return bytes(out)


def reference(packed,q8,n=N,ignore_min=False):
    need(len(packed)==n*(K//256)*Q4_BYTES and len(q8)==K//32*Q8_BYTES,'packed byte geometry')
    dots=[];bounds=[];amplitudes=[];corrections=[]
    u=2.0**-24;gamma=ROUNDING_CAP*u/(1-ROUNDING_CAP*u)
    for row in range(n):
        total=0.;amplitude=0.;correction=0.
        for block in range(K//256):
            off=(row*K//256+block)*Q4_BYTES;data=packed[off:off+Q4_BYTES]
            d,dm=struct.unpack_from('<ee',data);scales=data[4:16]
            for group in range(8):
                sc,mn=decode_scales(scales,group)
                q8off=(block*8+group)*Q8_BYTES;d8,s8=struct.unpack_from('<ee',q8,q8off)
                q=struct.unpack_from('<32b',q8,q8off+4)
                need(d8*sum(q)==s8,'independent Q8 sum identity')
                subtotal=0;unsigned_amp=0
                for j,v in enumerate(q):
                    byte=data[16+(group//2)*32+j];q4=(byte>>(4*(group%2)))&15
                    subtotal+=q4*v;unsigned_amp+=abs(q4*v)
                positive=d*sc*d8*subtotal;negative=dm*mn*d8*sum(q)
                total+=positive-(0 if ignore_min else negative)
                amplitude+=abs(d*sc*d8)*unsigned_amp+abs(dm*mn*d8)*sum(map(abs,q))
                correction+=negative
        # All values are dyadic with small bounded integer numerators: binary64 sum exact here.
        need(math.isfinite(total) and amplitude>0 and correction>0,'nontrivial row required')
        bound=gamma*amplitude
        dots.append(total);bounds.append(bound);amplitudes.append(amplitude);corrections.append(correction)
    return {'values':dots,'bounds':bounds,'unsigned_amplitude':amplitudes,'min_correction':corrections,
            'fp32_unit_roundoff':u,'rounding_cap':ROUNDING_CAP,'gamma':gamma}


def direct_reference(packed,input_bytes,n):
    """Separate elementwise dequantization, independent of grouped integer dot formulation."""
    x=struct.unpack('<%df'%K,input_bytes);values=[]
    for row in range(n):
        terms=[]
        for i in range(K):
            block,i256=divmod(i,256);group,j=divmod(i256,32)
            off=(row*K//256+block)*144
            d,dm=struct.unpack_from('<ee',packed,off)
            sc,mn=decode_scales(packed[off+4:off+16],group)
            byte=packed[off+16+(group//2)*32+j];q4=(byte>>(4*(group%2)))&15
            terms.append((d*sc*q4-dm*mn)*x[i])
        values.append(math.fsum(terms))
    return values


def file_ref(path):
    path=Path(path).resolve();data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def create_fixture():
    directory=HERE/'fixture';directory.mkdir(exist_ok=False)
    weights=make_weights();x=make_input();q8=expected_q8(x);r=reference(weights,q8)
    for name,data in [('weights.q4_k.bin',weights),('input.f32.bin',x),('expected.q8_1.bin',q8),
       ('reference.f64.bin',struct.pack('<%dd'%N,*r['values'])),('bounds.f64.bin',struct.pack('<%dd'%N,*r['bounds'])),
       ('amplitude.f64.bin',struct.pack('<%dd'%N,*r['unsigned_amplitude'])),('min_correction.f64.bin',struct.pack('<%dd'%N,*r['min_correction']))]:
        with (directory/name).open('xb') as f:f.write(data)
    payload={'schema':'q4k-dyadic-fixture/v1','M':1,'K':K,'N':N,'packed_Q4_block_bytes':144,'Q8_block_bytes':36,
      'files':{p.name:file_ref(p) for p in directory.iterdir()},'rounding_cap':ROUNDING_CAP,'fp32_unit_roundoff':2**-24,
      'bound':'gamma_(2K+64) * sum(abs(d*scale*q4*d8*q8) + abs(dmin*min*d8*q8)); no fitted tolerance',
      'CPU_reference_implementation':'independent Python packed six-bit scales/min and int dots; no ggml quantizer or dequantizer',
      'Q8_fixture':'nonzero signed dyadics; per32 max=127*d, d and sum exactly binary16, q integer exactly',
      'bounds_derivation':'integer DP4A and int scale/min products <2^24 so exact; at most128 partial dots (8 superblocks*16), local multiply/add paths and total reductions bounded conservatively by4160 FP32 roundings; cancellation uses unsigned amplitude',
      'scope':'only this nonzero synthetic fixture; no proof for arbitrary tensors, cache/layout generalization, LLM or cost'}
    with (directory/'manifest.json').open('x') as f:json.dump(payload,f,indent=2)
    return payload


if __name__=='__main__':print(json.dumps(create_fixture()))

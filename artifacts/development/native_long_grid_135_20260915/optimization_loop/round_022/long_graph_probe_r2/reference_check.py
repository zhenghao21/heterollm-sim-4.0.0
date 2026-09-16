"""Independent small host-only exact alternating SCALE reference check."""
import math,struct

def check():
    count=0
    for nodes in (64,256):
        for i in range(255):
            numerator=((i*73+19)%255)-127;x=numerator/256.0
            for stage in range(1,nodes+1):
                x*=0.5 if stage%2 else 2.0
                expected=math.ldexp(numerator,-8-stage%2)
                assert struct.pack('<f',x)==struct.pack('<f',expected) and math.isfinite(x)
                if numerator:assert x!=0
                count+=1
    return {'pass':True,'checked_stage_elements':count,'gpu_access':False}
if __name__=='__main__':print(check())

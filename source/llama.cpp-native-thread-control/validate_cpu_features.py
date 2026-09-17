from pathlib import Path
import ctypes,json,os,sys
folder=Path(sys.argv[1]).resolve();cookie=os.add_dll_directory(str(folder))
base=ctypes.CDLL(str(folder/'ggml-base.dll'));cpu=ctypes.CDLL(str(folder/'ggml-cpu.dll'))
cpu.ggml_backend_cpu_reg.restype=ctypes.c_void_p;reg=cpu.ggml_backend_cpu_reg()
base.ggml_backend_reg_get_proc_address.argtypes=[ctypes.c_void_p,ctypes.c_char_p];base.ggml_backend_reg_get_proc_address.restype=ctypes.c_void_p
address=base.ggml_backend_reg_get_proc_address(reg,b'ggml_backend_get_features')
class Feature(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('value',ctypes.c_char_p)]
get=ctypes.CFUNCTYPE(ctypes.POINTER(Feature),ctypes.c_void_p)(address);p=get(reg);result={};i=0
while p[i].name:
 result[p[i].name.decode()]=p[i].value.decode();i+=1
print(json.dumps(result,sort_keys=True))

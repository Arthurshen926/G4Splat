"""Extract a non-resumable render state without loading millions of candidate storages.

For trusted project-produced checkpoints only, like torch.load itself. Tensor
storages are deferred until top-level training-only state has been discarded.
No learned tensor values are changed. The input is never overwritten.
"""
import argparse
import io
import os
from pathlib import Path
import pickle
import tempfile
import time
import torch


class StorageRef:
    __slots__=("key","dtype","numel")
    def __init__(self,key,dtype,numel): self.key,self.dtype,self.numel=key,dtype,numel


class TensorRef:
    __slots__=("storage","offset","size","stride","requires_grad")
    def __init__(self,storage,offset,size,stride,requires_grad=False,*extra):
        self.storage,self.offset,self.size,self.stride=storage,offset,size,stride
        self.requires_grad=requires_grad
        if len(extra)>1 and extra[1]:
            raise ValueError("Tensor metadata needs an explicit extraction implementation")


class ParameterRef:
    __slots__=("tensor","requires_grad")
    def __init__(self,tensor,requires_grad,*extra):
        self.tensor,self.requires_grad=tensor,requires_grad
        if len(extra)>1 and extra[1]: raise ValueError("Parameter extra state is unsupported")


class DeferredUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=="torch._utils":
            if name in ("_rebuild_tensor","_rebuild_tensor_v2"): return TensorRef
            if name in ("_rebuild_parameter","_rebuild_parameter_with_state"): return ParameterRef
            if name.startswith("_rebuild_"):
                raise ValueError(f"Unsupported deferred tensor rebuild: {name}")
        if isinstance(name,str) and "Storage" in name:
            try: return torch.serialization.StorageType(name)
            except KeyError: pass
        return super().find_class(module,name)

    def persistent_load(self,identifier):
        if identifier[0] not in ("storage",b"storage"):
            raise ValueError("Unsupported persistent checkpoint record")
        storage_type,key,location,numel=identifier[1:]
        dtype=torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
        return StorageRef(str(key),dtype,int(numel))


def extract_state(source):
    source=Path(source)
    before=source.stat()
    reader=torch._C.PyTorchFileReader(str(source))
    state=DeferredUnpickler(io.BytesIO(reader.get_record("data.pkl"))).load()
    omitted=[]
    for key in ("static_ray_birth_state","volume_optimizer","volume_stats",
                "replacement_stats","foliage_ray_runtime_state","chart_atlas_optimizer"):
        if key in state: del state[key];omitted.append(key)
    cache={}
    loaded_bytes=0
    def materialize(value):
        nonlocal loaded_bytes
        if isinstance(value,TensorRef):
            storage=value.storage
            if storage.key not in cache:
                data=reader.get_record("data/"+storage.key)
                expected=storage.numel*torch._utils._element_size(storage.dtype)
                if len(data)!=expected: raise ValueError("Storage byte count differs from the checkpoint descriptor")
                loaded_bytes+=len(data)
                cache[storage.key]=(torch.frombuffer(bytearray(data),dtype=storage.dtype)
                                    if data else torch.empty(0,dtype=storage.dtype))
            tensor=torch.as_strided(cache[storage.key],value.size,value.stride,value.offset)
            return tensor.requires_grad_(bool(value.requires_grad))
        if isinstance(value,ParameterRef):
            return torch.nn.Parameter(materialize(value.tensor),requires_grad=value.requires_grad)
        if isinstance(value,dict): return {k:materialize(v) for k,v in value.items()}
        if isinstance(value,list): return [materialize(v) for v in value]
        if isinstance(value,tuple): return tuple(materialize(v) for v in value)
        return value
    state=materialize(state)
    after=source.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError("Source checkpoint changed during extraction")
    state["diagnostic_only"]=True
    state["checkpoint_kind"]="render_state_only_nonresumable"
    state["render_state_extraction"]={"source":str(source.resolve()),"source_bytes":before.st_size,
        "source_mtime_ns":before.st_mtime_ns,"omitted_training_fields":omitted,
        "loaded_storage_records":len(cache),"loaded_storage_bytes":loaded_bytes,
        "learned_tensor_values_unchanged":True}
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-iteration",type=int,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    start=time.monotonic();torch.set_num_threads(4)
    state=extract_state(args.input)
    if state.get("iteration")!=args.expected_iteration: raise ValueError("Unexpected checkpoint iteration")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=".render-state-",dir=args.output.parent)
    try:
        with os.fdopen(fd,"wb") as stream: torch.save(state,stream)
        os.link(temporary,args.output)
    finally: os.unlink(temporary)
    print({"output":str(args.output),"elapsed_sec":time.monotonic()-start,
           "bytes":args.output.stat().st_size,**state["render_state_extraction"]},flush=True)


if __name__=="__main__": main()

"""Create an immutable, numerically lossless checkpoint I/O copy."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from outdoor.checkpoint_encoding import compact_static_birth_vectors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-iteration",type=int,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    start=time.monotonic()
    digest=hashlib.sha256()
    with args.input.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""): digest.update(block)
    state=torch.load(args.input,map_location="cpu")
    if state.get("iteration")!=args.expected_iteration:
        raise ValueError(f"Expected iteration {args.expected_iteration}, found {state.get('iteration')}")
    state["static_ray_birth_state"]=compact_static_birth_vectors(state["static_ray_birth_state"])
    state["checkpoint_io_encoding"]={"policy":"lossless_static_birth_three_vector_lists_v1",
        "source":str(args.input.resolve()),"source_sha256":digest.hexdigest(),
        "model_optimizer_rng_schedule_and_training_contract_unchanged":True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=".compact-checkpoint-",dir=args.output.parent)
    os.close(fd)
    try:
        with open(temporary,"wb") as stream:
            torch.save(state,stream)
        # Atomic creation, never overwrite an existing output.
        os.link(temporary,args.output)
    finally:
        os.unlink(temporary)
    print({"output":str(args.output),"iteration":state["iteration"],
           "elapsed_sec":time.monotonic()-start,"bytes":args.output.stat().st_size},flush=True)


if __name__=="__main__": main()

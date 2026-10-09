#!/usr/bin/env python3
"""Quick GGUF tensor inspection — find vision-related tensors."""
import struct, os, sys

def read_gguf_tensor_names(path, max_tensors=2000):
    """Read tensor names from GGUF file."""
    names = []
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return ["NOT_GGUF"]
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        
        # Skip metadata KV pairs
        for _ in range(metadata_kv_count):
            # key
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")
            # value type
            vtype = struct.unpack("<I", f.read(4))[0]
            # skip value based on type
            if vtype == 0:  # UINT8
                f.read(1)
            elif vtype == 1:  # INT8
                f.read(1)
            elif vtype == 2:  # UINT16
                f.read(2)
            elif vtype == 3:  # INT16
                f.read(2)
            elif vtype == 4:  # UINT32
                f.read(4)
            elif vtype == 5:  # INT32
                f.read(4)
            elif vtype == 6:  # FLOAT32
                f.read(4)
            elif vtype == 7:  # BOOL
                f.read(1)
            elif vtype == 8:  # STRING
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
            elif vtype == 9:  # ARRAY
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                for _ in range(alen):
                    if atype == 8:  # STRING array
                        slen = struct.unpack("<Q", f.read(8))[0]
                        f.read(slen)
                    elif atype in (0, 1, 7):
                        f.read(1)
                    elif atype in (2, 3):
                        f.read(2)
                    elif atype in (4, 5, 6):
                        f.read(4)
                    elif atype in (10, 12):
                        f.read(8)
                    elif atype == 11:
                        f.read(8)
                    else:
                        return names  # can't parse further
            elif vtype == 10:  # UINT64
                f.read(8)
            elif vtype == 11:  # INT64
                f.read(8)
            elif vtype == 12:  # FLOAT64
                f.read(8)
            else:
                return names  # unknown type
        
        # Read tensor info
        for i in range(min(tensor_count, max_tensors)):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8", errors="replace")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            names.append((name, dims, dtype))
    return names

# Find model files
blob_dir = "/usr/share/ollama/.ollama/models/blobs"
print("Scanning GGUF files...")
for f in sorted(os.listdir(blob_dir)):
    fp = os.path.join(blob_dir, f)
    if not os.path.isfile(fp):
        continue
    sz = os.path.getsize(fp)
    if sz < 1_000_000_000:  # skip tiny files
        continue
    
    sz_gib = sz / (1024**3)
    # Check if GGUF
    with open(fp, "rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        continue
    
    print(f"\n{'='*60}")
    print(f"File: ...{f[-20:]} ({sz_gib:.2f} GiB)")
    
    try:
        tensors = read_gguf_tensor_names(fp)
        total = len(tensors) if isinstance(tensors, list) else 0
        print(f"Total tensors: {total}")
        
        # Find vision-related tensors
        vision_tensors = []
        for t in tensors:
            if isinstance(t, tuple):
                name, dims, dtype = t
                if any(k in name.lower() for k in ["vision", "visual", "image", "patch", "projector", "vit", "clip"]):
                    vision_tensors.append((name, dims, dtype))
        
        if vision_tensors:
            print(f"  VISION TENSORS FOUND: {len(vision_tensors)}")
            for name, dims, dtype in vision_tensors[:20]:
                print(f"    {name}: dims={dims} dtype={dtype}")
        else:
            print(f"  No vision tensors found")
        
        # Show first and last few tensor names for identification
        if tensors and isinstance(tensors[0], tuple):
            print(f"  First 3: {[t[0] for t in tensors[:3]]}")
            print(f"  Last 3:  {[t[0] for t in tensors[-3:]]}")
    except Exception as e:
        print(f"  Error: {e}")

import os
import json
import numpy as np

class DataFactory:
    """
    Responsible for generating, staging binary data files, and 
    calculating the Ground Truth (reference sum) for validation.
    """
    def __init__(self, workspace_dir: str = "temp_workspace"):
        self.workspace_dir = workspace_dir
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

        self.type_map = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float64": np.float64,
            "double": np.float64
        }

    def generate_data(self, size: int, dtype_str: str, mode: str) -> tuple[str, float]:
        # PARSING TYPE AND PRECISION (e.g., float32_3)
        parts = dtype_str.split("_")
        base_type = parts[0]
        decimals = None
        
        if len(parts) > 1 and parts[1].isdigit():
            decimals = int(parts[1])

        if base_type not in self.type_map:
            raise ValueError(f"[DataFactory] Unsupported base data type: {base_type}")

        if "int" in base_type and decimals is not None:
            raise ValueError(f"[DataFactory] Cannot apply decimal precision to integer type: {dtype_str}")

        np_type = self.type_map[base_type]
        filename = f"dataset_{mode}_{dtype_str}_{size}.bin"
        filepath = os.path.join(self.workspace_dir, filename)
        meta_filepath = filepath + ".meta"

        if os.path.exists(filepath) and os.path.exists(meta_filepath):
            with open(meta_filepath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[DataFactory] Reusing existing dataset: {filename}")
            return os.path.abspath(filepath), float(meta.get("reference_sum", 0.0))

        print(f"[DataFactory] Generating new MASSIVE dataset: {filename}...")
        
        chunk_size = 100_000_000 
        elements_written = 0
        reference_sum = 0.0

        # PARSING BINARY FRACTION ALIGNMENT
        is_binary_aligned = "_binary" in mode or mode == "binary"
        base_mode = mode.replace("_binary", "").replace("binary", "")
        
        if not base_mode:
            base_mode = "random_uniform"

        # PARSING BOUNDS (e.g., "0+-5")
        low_bound, high_bound = 0.0, 1.0
        is_custom_bounds = False

        if base_mode.startswith("0+-"):
            try:
                val = float(base_mode.split("+-")[1])
                low_bound = -val
                high_bound = val
                is_custom_bounds = True
            except ValueError:
                raise ValueError(f"[DataFactory] Invalid bounds format: {mode}")

        with open(filepath, "wb") as f:
            while elements_written < size:
                current_chunk_size = min(chunk_size, size - elements_written)
                
                if base_mode == "zeros":
                    data = np.zeros(current_chunk_size, dtype=np_type)
                elif base_mode == "ones":
                    data = np.ones(current_chunk_size, dtype=np_type)
                elif base_mode == "sequential":
                    data = np.arange(elements_written, elements_written + current_chunk_size, dtype=np_type)
                elif base_mode == "random_uniform" or is_custom_bounds:
                    if not is_custom_bounds:
                        if "int" in base_type:
                            low_bound, high_bound = 0, 10
                        else:
                            low_bound, high_bound = 0.0, 1.0
                    
                    if "int" in base_type:
                        data = np.random.randint(int(low_bound), int(high_bound) + 1, size=current_chunk_size, dtype=np_type)
                    else:
                        data = np.random.uniform(low_bound, high_bound, current_chunk_size).astype(np_type)
                        
                        # APPLY BASE-2 BINARY FRACTIONS (e.g. 1/8, 1/16)
                        if is_binary_aligned:
                            if decimals is None:
                                raise ValueError(f"[DataFactory] Binary mode requires explicit precision in type (e.g., float32_3).")
                            step = 2.0 ** (-decimals)
                            data = np.round(data / step) * step
                        # APPLY STANDARD BASE-10 DECIMALS
                        elif decimals is not None:
                            data = np.round(data, decimals=decimals)
                            
                        data = data.astype(np_type)
                else:
                    raise ValueError(f"[DataFactory] Unsupported generation mode: {mode}")

                reference_sum += float(np.sum(data, dtype=np.float64))
                
                f.write(data.tobytes())
                elements_written += current_chunk_size
                
                if size >= 100_000_000:
                    print(f"  -> Generated and written {elements_written}/{size} elements...")
        
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump({"reference_sum": reference_sum}, f)

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"[DataFactory] Saved {file_size_gb:.2f} GB and calculated Ground Truth.")
        
        return os.path.abspath(filepath), reference_sum
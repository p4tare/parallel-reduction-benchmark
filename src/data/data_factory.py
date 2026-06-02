import os
import numpy as np

class DataFactory:
    """
    Responsible for generating and staging binary data files.
    These files are later read by the C++/CUDA algorithms to ensure
    data generation overhead is excluded from the performance measurements.
    """
    def __init__(self, workspace_dir: str = "temp_workspace"):
        self.workspace_dir = workspace_dir
        
        # Ensure the temporary workspace exists
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

        # Mapping string types from YAML to numpy dtypes
        self.type_map = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32,
            "float64": np.float64,
            "double": np.float64  # alias for float64
        }

    def generate_data(self, size: int, dtype_str: str, mode: str) -> str:
        """
        Generates an array and saves it as a raw binary file.
        Returns the absolute path to the generated file.
        """
        if dtype_str not in self.type_map:
            raise ValueError(f"Unsupported data type: {dtype_str}")

        np_type = self.type_map[dtype_str]
        
        # Create a unique filename for this configuration
        filename = f"dataset_{mode}_{dtype_str}_{size}.bin"
        filepath = os.path.join(self.workspace_dir, filename)

        # If file already exists, skip generation to save time
        if os.path.exists(filepath):
            print(f"[DataFactory] Reusing existing dataset: {filename}")
            return os.path.abspath(filepath)

        print(f"[DataFactory] Generating new dataset: {filename}...")
        
        # Generate data based on the requested mode
        if mode == "random_uniform":
            if "int" in dtype_str:
                # Random integers between 0 and 1000
                data = np.random.randint(0, 100, size=size, dtype=np_type)
            else:
                # Random floats between 0.0 and 1.0
                data = np.random.rand(size).astype(np_type)
                
        elif mode == "zeros":
            data = np.zeros(size, dtype=np_type)
            
        elif mode == "sequential":
            data = np.arange(size, dtype=np_type)
            
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        # Save to raw binary format (no headers, purely bytes)
        data.tofile(filepath)
        
        file_size_mb = data.nbytes / (1024 * 1024)
        print(f"[DataFactory] Saved {file_size_mb:.2f} MB to {filepath}")
        
        return os.path.abspath(filepath)


# Execution block for testing
if __name__ == "__main__":
    factory = DataFactory()
    
    print("Testing Data Factory...")
    
    # Test 1: float32, 1 million elements, random
    path1 = factory.generate_data(1000000, "float32", "random_uniform")
    
    # Test 2: int32, 5 million elements, sequential
    path2 = factory.generate_data(5000000, "int32", "sequential")
    
    print("\nData staging complete. Files ready for C++:")
    print(f" - {path1}")
    print(f" - {path2}")
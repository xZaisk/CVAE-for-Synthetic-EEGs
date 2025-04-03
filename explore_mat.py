import scipy.io
import numpy as np
import pprint

# Load one file
file_path = "bci_iv_2a_data/A01T.mat"  # Update with your actual file path
mat_data = scipy.io.loadmat(file_path)

# Print the top-level keys in the file
print("Top-level keys:")
print(list(mat_data.keys()))

# Function to recursively explore nested structures
def explore_structure(data, name="", level=0):
    indent = "  " * level
    
    if isinstance(data, np.ndarray):
        if data.dtype.kind == 'O':  # If it contains Python objects
            print(f"{indent}{name}: ndarray of objects, shape {data.shape}")
            
            # If it's not too large, explore the elements
            if data.size <= 10:
                for i, element in enumerate(data.flat):
                    explore_structure(element, f"{name}[{i}]", level + 1)
            else:
                print(f"{indent}  (showing only first element due to size)")
                explore_structure(data.flat[0], f"{name}[0]", level + 1)
                
        elif data.dtype.names is not None:  # Structured array
            print(f"{indent}{name}: structured ndarray with fields {data.dtype.names}, shape {data.shape}")
            
            # Show one element as example
            if data.size > 0:
                for field in data.dtype.names:
                    explore_structure(data[field][0], f"{name}[0].{field}", level + 1)
                    
        else:  # Regular array
            print(f"{indent}{name}: ndarray of {data.dtype}, shape {data.shape}")
            
            # For small arrays, show some values
            if data.size <= 20:
                print(f"{indent}  values: {data}")
            
    elif isinstance(data, dict):
        print(f"{indent}{name}: dict with keys {list(data.keys())}")
        for key, value in data.items():
            if key not in ['__header__', '__version__', '__globals__']:
                explore_structure(value, f"{name}['{key}']", level + 1)
                
    elif hasattr(data, '__dict__'):  # For custom objects
        print(f"{indent}{name}: object with attributes {list(data.__dict__.keys())}")
        for attr, value in data.__dict__.items():
            explore_structure(value, f"{name}.{attr}", level + 1)
            
    else:  # Basic types
        print(f"{indent}{name}: {type(data).__name__}, value: {data}")

# Explore the structure of the file
print("\nDetailed structure:")
for key in mat_data.keys():
    if key not in ['__header__', '__version__', '__globals__']:
        explore_structure(mat_data[key], key)
import json
import math

def compare_json(obj1, obj2, path="", tol=1e-6):
    differences = []
    
    # Both objects are dictionaries.
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())
        if keys1 != keys2:
            missing_in_obj2 = keys1 - keys2
            missing_in_obj1 = keys2 - keys1
            if missing_in_obj2:
                differences.append(f"{path}: Keys missing in second JSON: {missing_in_obj2}")
            if missing_in_obj1:
                differences.append(f"{path}: Keys missing in first JSON: {missing_in_obj1}")
        # Compare shared keys.
        for key in keys1 & keys2:
            new_path = f"{path}.{key}" if path else key
            differences.extend(compare_json(obj1[key], obj2[key], new_path, tol))
    
    # Both objects are lists.
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: List lengths differ ({len(obj1)} vs {len(obj2)})")
        # Compare each pair of items (only up to the length of the shorter list).
        for index, (item1, item2) in enumerate(zip(obj1, obj2)):
            new_path = f"{path}[{index}]"
            differences.extend(compare_json(item1, item2, new_path, tol))
    
    # Both objects are floats.
    elif isinstance(obj1, float) and isinstance(obj2, float):
        if not math.isclose(obj1, obj2, abs_tol=tol):
            differences.append(f"{path}: Floats differ ({obj1} vs {obj2})")
    
    # For other types (int, str, bool, None), use direct comparison.
    else:
        if obj1 != obj2:
            differences.append(f"{path}: Values differ ({obj1} vs {obj2})")
    
    return differences

def main():
    with open('output.json', 'r') as f1:
        data1 = json.load(f1)
    with open('output1.json', 'r') as f2:
        data2 = json.load(f2)
    
    differences = compare_json(data1, data2)
    
    if differences:
        print("JSON files differ:")
        for diff in differences:
            print(" -", diff)
    else:
        print("JSON files are the same.")

if __name__ == '__main__':
    main()

import os
import ast
import importlib
import sys

def find_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=file_path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.level > 0:
                    # skip relative imports
                    continue
                imports.append(node.module)
    return imports

def validate_imports(project_path):
    sys.path.insert(0, project_path)  # allow imports relative to root
    errors = {}
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = find_imports_in_file(file_path)
                for module in imports:
                    try:
                        importlib.import_module(module)
                    except ModuleNotFoundError:
                        rel_file = os.path.relpath(file_path, project_path)
                        errors.setdefault(rel_file, []).append(module)
                    except Exception as e:
                        print(f"⚠️ Unexpected error in {file}: {e}")
    return errors

if __name__ == "__main__":
    project_path = "/Volumes/MacMiniExt/Personal/Lotto/lotto_predictor_new"  # change if needed
    missing_imports = validate_imports(project_path)
    if missing_imports:
        print("❌ Missing or invalid imports found:\n")
        for file, modules in missing_imports.items():
            print(f"In {file}:")
            for mod in modules:
                print(f"  - {mod}")
    else:
        print("✅ All imports are resolvable from Python's perspective.")

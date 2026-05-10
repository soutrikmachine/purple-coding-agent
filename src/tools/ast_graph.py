"""
Note: This module implements a lightweight Graph RAG using Tree-sitter.
It extracts the structural skeleton (classes, functions, and imports) of Python and JS codebases.
"""

import os
import sys
import logging
from typing import Dict, List, Optional

# Import Tree-sitter and the language grammars
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript

logger = logging.getLogger(__name__)

class ASTGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
        # --- 1. Load Languages ---
        self.PY_LANGUAGE = Language(tspython.language())
        self.JS_LANGUAGE = Language(tsjavascript.language())
        
        # --- 2. Initialize Parsers ---
        self.py_parser = Parser(self.PY_LANGUAGE)
        self.js_parser = Parser(self.JS_LANGUAGE)
        
        # --- 3. Define Python Queries ---
        self.py_query = self.PY_LANGUAGE.query("""
            (import_statement) @import
            (import_from_statement) @import_from
            (class_definition name: (identifier) @class_name) @class
            (function_definition name: (identifier) @func_name) @function
        """)

        # --- 4. Define JavaScript Queries ---
        # JS has different AST node types (e.g., arrow functions, class declarations)
        self.js_query = self.JS_LANGUAGE.query("""
            (import_statement) @import
            (class_declaration name: (identifier) @class_name) @class
            (function_declaration name: (identifier) @func_name) @function
            (variable_declarator name: (identifier) @func_name value: [(arrow_function) (function_expression)]) @function
            (method_definition name: (property_identifier) @func_name) @function
        """)

    def _read_file(self, file_path: str) -> Optional[bytes]:
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None

    def parse_file_skeleton(self, file_path: str, ext: str) -> Dict[str, List[str]]:
        source_code = self._read_file(file_path)
        if not source_code:
            return {"imports": [], "classes": [], "functions": []}

        # Route to the correct parser and query
        if ext == '.py':
            tree = self.py_parser.parse(source_code)
            captures = self.py_query.captures(tree.root_node)
        elif ext in ['.js', '.jsx']:
            tree = self.js_parser.parse(source_code)
            captures = self.js_query.captures(tree.root_node)
        else:
            return {"imports": [], "classes": [], "functions": []}

        skeleton = {"imports": [], "classes": [], "functions": []}

        def get_text(node: Node) -> str:
            return source_code[node.start_byte:node.end_byte].decode('utf-8')

        for node, capture_name in captures.items():
            if capture_name in ["import", "import_from"]:
                # Limit import text length to avoid massive inline require strings
                import_text = get_text(node).strip()
                skeleton["imports"].append(import_text[:100] + "..." if len(import_text) > 100 else import_text)
            elif capture_name == "class_name":
                skeleton["classes"].append(get_text(node))
            elif capture_name == "func_name":
                skeleton["functions"].append(get_text(node))

        # Deduplicate JS arrow functions that might get captured twice
        skeleton["classes"] = list(dict.fromkeys(skeleton["classes"]))
        skeleton["functions"] = list(dict.fromkeys(skeleton["functions"]))

        return skeleton

    def build_repo_graph(self, exclude_dirs: Optional[List[str]] = None) -> str:
        if exclude_dirs is None:
            exclude_dirs = ['.git', '__pycache__', 'venv', 'env', 'node_modules', 'tests', 'vendor', 'dist', 'build']

        graph_output = ["# Repository Architecture Skeleton\n"]

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                # Accept both Python and JS files
                ext = os.path.splitext(file)[1].lower()
                if ext not in ['.py', '.js', '.jsx']:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.repo_path)
                skeleton = self.parse_file_skeleton(full_path, ext)
                
                if skeleton["classes"] or skeleton["functions"]:
                    graph_output.append(f"## File: `{rel_path}`")
                    if skeleton["classes"]:
                        graph_output.append("  **Classes:**")
                        for cls in skeleton["classes"]:
                            graph_output.append(f"   - {cls}")
                    if skeleton["functions"]:
                        graph_output.append("  **Functions:**")
                        for func in skeleton["functions"]:
                            graph_output.append(f"   - {func}")
                    graph_output.append("")

        return "\n".join(graph_output)


# --- EXECUTABLE BASH CLI INTERFACE ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/tools/ast_graph.py <TargetName>")
        sys.exit(1)
        
    target = sys.argv[1]
    
    # Use the current directory as the workspace root unless an env var is set
    workspace_dir = os.environ.get("WORKSPACE_DIR", ".") 
    
    print(f"Building AST and searching for: '{target}'...")
    
    try:
        builder = ASTGraphBuilder(repo_path=workspace_dir)
        full_skeleton = builder.build_repo_graph()
        
        # Filter the massive graph to only show where the target lives
        found = False
        current_file = ""
        for line in full_skeleton.split('\n'):
            if line.startswith("## File:"):
                current_file = line
            elif target in line:
                if not found:
                    print("\n--- AST SEARCH RESULTS ---")
                print(current_file)
                print(line)
                found = True
                
        if not found:
            print(f"Could not find '{target}' in any parsed Python or JavaScript files.")
            print("Try using standard grep if it is a variable name rather than a function/class.")
            
    except Exception as e:
        print(f"AST Graph Builder Error: {e}")
        sys.exit(1)
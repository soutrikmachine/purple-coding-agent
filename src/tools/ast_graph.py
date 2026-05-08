"""
Note: This module implements a lightweight Graph RAG using Tree-sitter.
It extracts the structural skeleton (classes, functions, and imports) of a Python codebase.
This drastically reduces token usage and provides the LLM with an immediate, high-level 
map of the repository architecture without the overhead of full-text embeddings.
"""

import os
import logging
from typing import Dict, List, Optional

# Requires: pip install tree-sitter tree-sitter-python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

logger = logging.getLogger(__name__)

class ASTGraphBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
        # Initialize the Tree-sitter parser for Python
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        
        # Define queries to extract the skeleton
        self.skeleton_query = self.PY_LANGUAGE.query("""
            (import_statement) @import
            (import_from_statement) @import_from
            (class_definition name: (identifier) @class_name) @class
            (function_definition name: (identifier) @func_name) @function
        """)

    def _read_file(self, file_path: str) -> Optional[bytes]:
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None

    def parse_file_skeleton(self, file_path: str) -> Dict[str, List[str]]:
        """
        Parses a single Python file and returns its structural components.
        """
        source_code = self._read_file(file_path)
        if not source_code:
            return {"imports": [], "classes": [], "functions": []}

        tree = self.parser.parse(source_code)
        captures = self.skeleton_query.captures(tree.root_node)

        skeleton = {
            "imports": [],
            "classes": [],
            "functions": []
        }

        # Helper to extract text from a node
        def get_text(node: Node) -> str:
            return source_code[node.start_byte:node.end_byte].decode('utf-8')

        for node, capture_name in captures.items():
            if capture_name in ["import", "import_from"]:
                skeleton["imports"].append(get_text(node).strip())
            elif capture_name == "class_name":
                skeleton["classes"].append(get_text(node))
            elif capture_name == "func_name":
                skeleton["functions"].append(get_text(node))

        return skeleton

    def build_repo_graph(self, exclude_dirs: Optional[List[str]] = None) -> str:
        """
        Walks the repository and builds a condensed markdown representation 
        of the entire codebase structure to be injected into the LLM context.
        """
        if exclude_dirs is None:
            exclude_dirs = ['.git', '__pycache__', 'venv', 'env', 'node_modules', 'tests']

        graph_output = ["# Repository Architecture Skeleton\n"]

        for root, dirs, files in os.walk(self.repo_path):
            # Mutate dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith('.py'):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.repo_path)
                
                skeleton = self.parse_file_skeleton(full_path)
                
                # Only include files that actually have structure
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
                            
                    graph_output.append("") # Blank line for readability

        return "\n".join(graph_output)
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ICLSpecialist:
    """
    Stage 2.5: Dynamic Prompt Injection.
    Injects high-quality In-Context Learning examples and domain-specific 
    knowledge based on the detected environment and issue type.
    """
    def __init__(self):
        # Library of domain-specific coding patterns
        self.knowledge_base = {
            "django": [
                "When fixing Django models, ensure you run 'python manage.py makemigrations' if fields change.",
                "QuerySets are lazy; use .select_related() if the issue involves N+1 query performance.",
                "Example Fix: Changing a model field often requires updating the admin.py registration."
            ],
            "pytest": [
                "Use 'pytest -k <keyword>' to run specific failing tests.",
                "Mocking: Use the 'mocker' fixture from pytest-mock rather than the standard unittest.mock.",
                "If a test fails with OOM, check for unclosed database connections in fixtures."
            ],
            "general_swe": [
                "Always check for off-by-one errors in loop boundaries.",
                "Ensure edge cases like empty strings, None, or zero are handled in new logic.",
                "When patching, copy the exact indentation of the surrounding context lines."
            ]
        }

    def detect_domain(self, problem_statement: str, repo_skeleton: str) -> List[str]:
        """Detects the framework or library being used."""
        domains = ["general_swe"]
        ps_lower = problem_statement.lower()
        sk_lower = repo_skeleton.lower()
        
        if "django" in ps_lower or "manage.py" in sk_lower:
            domains.append("django")
        if "pytest" in ps_lower or "conftest.py" in sk_lower:
            domains.append("pytest")
            
        return domains

    def get_injection(self, problem_statement: str, repo_skeleton: str) -> str:
        """Constructs the ICL block to be injected into the system prompt."""
        domains = self.detect_domain(problem_statement, repo_skeleton)
        
        injection_lines = [
            "\n## Domain-Specific Expert Knowledge",
            "Follow these specialist guidelines for the detected environment:"
        ]
        
        for domain in domains:
            if domain in self.knowledge_base:
                for rule in self.knowledge_base[domain]:
                    injection_lines.append(f"- {rule}")
                    
        return "\n".join(injection_lines)

    def get_few_shot_examples(self) -> str:
        """Provides high-quality examples of the <thought>/<action> loop."""
        return (
            "\n## Example Turn Structure:\n"
            "<thought>I need to find where the validation logic for users is defined.</thought>\n"
            "<action type=\"bash\">grep -rn \"class User\" .</action>\n"
            "<observation>\n./models/auth.py:42:class User(models.Model):\n</observation>\n"
            "\n<thought>The user is reporting an error in the save method. I will read that file.</thought>\n"
            "<action type=\"bash\">cat -n models/auth.py | head -n 60 | tail -n 20</action>\n"
        )
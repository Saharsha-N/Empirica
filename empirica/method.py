import re
from pathlib import Path

from .key_manager import KeyManager
    # fix bug
from .prompts.method import method_planner_prompt, method_researcher_prompt
from .utils import create_work_dir, get_task_result
from .exceptions import TaskResultError, AgentExecutionError, ContentExtractionError
from .logger import get_logger

def _import_cmbagent():
    """Lazy import of cmbagent with helpful error message."""
    try:
        import cmbagent
        return cmbagent
    except ImportError:
        raise ImportError(
            "cmbagent is required for this functionality but is not installed. "
            "Please install it with: pip install cmbagent>=0.0.1post63"
        ) from None

logger = get_logger(__name__)

class Method:
    """
    Develops research project methodology based on data description and research idea.
    
    This class uses multi-agent systems to generate a detailed research methodology
    that outlines how to conduct the research project. The methodology is developed
    through a collaborative process involving planner and researcher agents.
    
    The generated methodology is returned as a formatted markdown document that can
    be used to guide the experimental execution.

    Args:
        research_idea: The research idea or hypothesis for which to develop methodology.
        keys: KeyManager instance containing API keys for LLM services.
        work_dir: Working directory where methodology generation outputs will be stored.
        researcher_model: LLM model identifier for the researcher agent. Defaults to "gpt-4.1-2025-04-14".
        planner_model: LLM model identifier for the planner agent. Defaults to "gpt-4.1-2025-04-14".
        plan_reviewer_model: LLM model identifier for the plan reviewer agent. Defaults to "o3-mini".
        orchestration_model: LLM model identifier for orchestration tasks. Defaults to "gpt-4.1".
        formatter_model: LLM model identifier for formatting results. Defaults to "o3-mini".
    """

    def __init__(self,
                 research_idea: str,
                 keys: KeyManager,
                 work_dir: str | Path,
                 researcher_model: str = "gpt-4.1-2025-04-14",
                 planner_model: str = "gpt-4.1-2025-04-14",
                 plan_reviewer_model: str = "o3-mini",
                 orchestration_model: str = "gpt-4.1",
                 formatter_model: str = "o3-mini",
                ):
        
        self.researcher_model = researcher_model
        self.planner_model = planner_model
        self.plan_reviewer_model = plan_reviewer_model
        self.orchestration_model = orchestration_model
        self.formatter_model = formatter_model
        self.api_keys = keys

        self.method_dir = create_work_dir(work_dir, "method")

        # Set prompts
        self.planner_append_instructions = method_planner_prompt.format(research_idea=research_idea)
        self.researcher_append_instructions = method_researcher_prompt.format(research_idea=research_idea)

    def develop_method(self, data_description: str) -> str:
        """
        Develop a research methodology based on the data description and research idea.
        
        This method orchestrates the methodology generation process using the configured
        agents. The methodology is extracted from a markdown code block in the agent's response
        and returned as a cleaned string.

        Args:
            data_description: Description of the data and tools available for the research.

        Returns:
            A string containing the developed research methodology in markdown format.
        
        Raises:
            AgentExecutionError: If the methodology generation process fails or the methodology
                cannot be extracted from the agent's response.
            ContentExtractionError: If the methodology cannot be extracted from the markdown
                code block format.
        """

        cmbagent = _import_cmbagent()
        results = cmbagent.planning_and_control_context_carryover(data_description,
                              n_plan_reviews = 1,
                              max_n_attempts = 4,
                              max_plan_steps = 4,
                              researcher_model = self.researcher_model,
                              planner_model = self.planner_model,
                              plan_reviewer_model = self.plan_reviewer_model,
                              plan_instructions = self.planner_append_instructions,
                              researcher_instructions = self.researcher_append_instructions,
                              work_dir = self.method_dir,
                              api_keys = self.api_keys,
                              default_llm_model = self.orchestration_model,
                              default_formatter_model = self.formatter_model
                             )
        
        chat_history = results['chat_history']
        
        try:
            task_result = get_task_result(chat_history, 'researcher_response_formatter')
        except TaskResultError as e:
            logger.error(f"Failed to extract method from chat history: {e}")
            raise AgentExecutionError(
                agent_name="researcher",
                stage="result_extraction",
                message=f"Could not extract method result from agent execution: {e}"
            ) from e
        except KeyError as e:
            logger.error(f"Chat history structure error: {e}")
            raise AgentExecutionError(
                agent_name="researcher",
                stage="result_extraction",
                message=f"Chat history has unexpected structure: {e}"
            ) from e
        
        MD_CODE_BLOCK_PATTERN = r"```[ \t]*(?:markdown)[ \t]*\r?\n(.*)\r?\n[ \t]*```"
        try:
            matches = re.findall(MD_CODE_BLOCK_PATTERN, task_result, flags=re.DOTALL)
            if not matches:
                raise ContentExtractionError(
                    content_type="methodology",
                    source="researcher_response_formatter",
                    message=(
                        f"Could not extract methodology from markdown code block. "
                        f"Expected format: ```markdown ... ```. "
                        f"Task result length: {len(task_result)} characters."
                    )
                )
            extracted_methodology = matches[0]
        except IndexError as e:
            logger.error(f"Failed to extract methodology from markdown block: {e}")
            raise ContentExtractionError(
                content_type="methodology",
                source="researcher_response_formatter",
                message=f"Failed to extract methodology from formatted response: {e}"
            ) from e
        
        clean_methodology = re.sub(r'^<!--.*?-->\s*\n', '', extracted_methodology)
        return clean_methodology

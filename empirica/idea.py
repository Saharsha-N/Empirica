import re
from pathlib import Path

from .key_manager import KeyManager
from .prompts.idea import idea_planner_prompt
from .utils import create_work_dir, get_task_result
from .exceptions import TaskResultError, AgentExecutionError
from .logger import get_logger
    # improve test coverage

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

class Idea:
    """
    This class is used to develop a research project idea based on the data of interest.
    It makes use of two types of agents:

    - `idea_maker`: to generate new ideas.
    - `idea_hater`: to critique new ideas.
    
    The LLMs are provided the following instructions:

    - Ask `idea_maker` to generate 5 new research project ideas related to the datasets.
    - Ask `idea_hater` to critique these ideas.
    - Ask `idea_maker` to select and improve 2 out of the 5 research project ideas given the output of the `idea_hater`.
    - Ask `idea_hater` to critique the 2 improved ideas. 
    - Ask `idea_maker` to select the best idea out of the 2. 
    - Ask `idea_maker` to report the best idea in the form of a scientific paper title with a 5-sentence description. 

    Args:
        keys: KeyManager instance containing API keys for LLM services.
        work_dir: Working directory where idea generation outputs will be stored.
        idea_maker_model: LLM model identifier for the idea_maker agent. Defaults to "gpt-4o".
        idea_hater_model: LLM model identifier for the idea_hater agent. Defaults to "o3-mini".
        planner_model: LLM model identifier for the planner agent. Defaults to "gpt-4o".
        plan_reviewer_model: LLM model identifier for the plan reviewer agent. Defaults to "o3-mini".
        orchestration_model: LLM model identifier for orchestration tasks. Defaults to "gpt-4.1".
        formatter_model: LLM model identifier for formatting results. Defaults to "o3-mini".
    """
    def __init__(self, 
                 keys: KeyManager,
                 work_dir: str | Path,
                 idea_maker_model: str = "gpt-4o", 
                 idea_hater_model: str = "o3-mini",
                 planner_model: str = "gpt-4o",
                 plan_reviewer_model: str = "o3-mini",
                 orchestration_model: str = "gpt-4.1",
                 formatter_model: str = "o3-mini",
                ):
        
        self.idea_maker_model = idea_maker_model
        self.idea_hater_model = idea_hater_model
        self.planner_model = planner_model
        self.plan_reviewer_model = plan_reviewer_model
        self.orchestration_model = orchestration_model
        self.formatter_model = formatter_model
        self.api_keys = keys

        self.idea_dir = create_work_dir(work_dir, "idea")

        # Set prompt
        self.planner_append_instructions = idea_planner_prompt
        
    def develop_idea(self, data_description: str) -> str:
        """
        Develop a research project idea based on the provided data description.
        
        This method orchestrates a multi-step idea generation process using the configured
        agents. The process involves generating multiple ideas, critiquing them, refining
        the best candidates, and ultimately selecting the best research idea.
        
        The final idea is returned as a formatted string containing a scientific paper title
        and a 5-sentence description.

        Args:
            data_description: Description of the data and tools available for research.

        Returns:
            A string containing the selected research idea in the format of a scientific
            paper title with a 5-sentence description.
        
        Raises:
            AgentExecutionError: If the idea generation process fails or the idea cannot
                be extracted from the agent's response.
        """
        
        cmbagent = _import_cmbagent()
        results = cmbagent.planning_and_control_context_carryover(data_description,
                              n_plan_reviews = 1,
                              max_plan_steps = 6,
                              idea_maker_model = self.idea_maker_model,
                              idea_hater_model = self.idea_hater_model,
                              plan_instructions=self.planner_append_instructions,
                              planner_model=self.planner_model,
                              plan_reviewer_model=self.plan_reviewer_model,
                              work_dir = self.idea_dir,
                              api_keys = self.api_keys,
                              default_llm_model = self.orchestration_model,
                              default_formatter_model = self.formatter_model
                             )

        chat_history = results['chat_history']
        
        try:
            task_result = get_task_result(chat_history, 'idea_maker_nest')
        except TaskResultError as e:
            logger.error(f"Failed to extract idea from chat history: {e}")
            raise AgentExecutionError(
                agent_name="idea_maker",
                stage="result_extraction",
                message=f"Could not extract idea result from agent execution: {e}"
            ) from e
        except KeyError as e:
            logger.error(f"Chat history structure error: {e}")
            raise AgentExecutionError(
                agent_name="idea_maker",
                stage="result_extraction",
                message=f"Chat history has unexpected structure: {e}"
            ) from e

        pattern = r'\*\*Ideas\*\*\s*\n- Idea 1:'
        replacement = "Project Idea:"
        task_result = re.sub(pattern, replacement, task_result)

        return task_result

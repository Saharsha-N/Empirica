import re
from pathlib import Path

from .key_manager import KeyManager
from .prompts.experiment import experiment_planner_prompt, experiment_engineer_prompt, experiment_researcher_prompt
from .utils import create_work_dir, get_task_result
    # improve code comments
    # update examples
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

class Experiment:
    """
    Orchestrates the execution of research experiments using multi-agent systems.
    
    This class coordinates the execution of experiments by leveraging the `cmbagent` backend
    for detailed planning and control. It uses multiple specialized agents (engineer, researcher,
    planner, and plan reviewer) to execute the research methodology and generate results.
    
    The experiment execution follows a structured workflow:
    1. Planning: The planner agent creates an execution plan based on the research idea and methodology
    2. Execution: Engineer and researcher agents execute the plan, with the researcher generating results
    3. Formatting: Results are extracted from markdown code blocks and stored for further use
    
    Results are stored in `self.results` and plot paths in `self.plot_paths` after execution.

    Args:
        research_idea: The research idea or hypothesis to be tested.
        methodology: The methodology describing how to conduct the experiment.
        keys: KeyManager instance containing API keys for LLM services.
        work_dir: Working directory where experiment outputs will be stored.
        involved_agents: List of agent types to involve in the experiment. Defaults to ['engineer', 'researcher'].
        engineer_model: LLM model identifier for the engineer agent. Defaults to "gpt-4.1".
        researcher_model: LLM model identifier for the researcher agent. Defaults to "o3-mini-2025-01-31".
        planner_model: LLM model identifier for the planner agent. Defaults to "gpt-4o".
        plan_reviewer_model: LLM model identifier for the plan reviewer agent. Defaults to "o3-mini".
        restart_at_step: Step number to restart from if resuming a previous experiment. Use -1 for new experiments.
        hardware_constraints: Optional string describing hardware constraints for the experiment.
        max_n_attempts: Maximum number of attempts allowed for plan execution. Defaults to 10.
        max_n_steps: Maximum number of steps in the execution plan. Defaults to 6.
        orchestration_model: LLM model identifier for orchestration tasks. Defaults to "gpt-4.1".
        formatter_model: LLM model identifier for formatting results. Defaults to "o3-mini".
    """

    def __init__(self,
                 research_idea: str,
                 methodology: str,
                 keys: KeyManager,
                 work_dir: str | Path,
                 involved_agents: list[str] = ['engineer', 'researcher'],
                 engineer_model: str = "gpt-4.1",
                 researcher_model: str = "o3-mini-2025-01-31",
                 planner_model: str = "gpt-4o",
                 plan_reviewer_model: str = "o3-mini",
                 restart_at_step: int = -1,
                 hardware_constraints: str | None = None,
                 max_n_attempts: int = 10,
                 max_n_steps: int = 6,
                 orchestration_model: str = "gpt-4.1",
                 formatter_model: str = "o3-mini",
                ):
        
        self.engineer_model = engineer_model
        self.researcher_model = researcher_model
        self.planner_model = planner_model
        self.plan_reviewer_model = plan_reviewer_model
        self.restart_at_step = restart_at_step
        if hardware_constraints is None:
            hardware_constraints = ""
        self.hardware_constraints = hardware_constraints
        self.max_n_attempts = max_n_attempts
        self.max_n_steps = max_n_steps
        self.orchestration_model = orchestration_model
        self.formatter_model = formatter_model

        self.api_keys = keys

        self.experiment_dir = create_work_dir(work_dir, "experiment")

        involved_agents_str = ', '.join(involved_agents)

        # Set prompts
        self.planner_append_instructions = experiment_planner_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
            involved_agents_str = involved_agents_str
        )
        self.engineer_append_instructions = experiment_engineer_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )
        self.researcher_append_instructions = experiment_researcher_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )

    def run_experiment(self, data_description: str, **kwargs) -> None:
        """
        Execute the experiment using the configured agents and methodology.
        
        This method orchestrates the full experiment execution workflow:
        1. Logs the experiment configuration
        2. Calls the cmbagent planning and control system with the configured agents
        3. Extracts results from the researcher agent's formatted response
        4. Parses markdown code blocks to extract the final results
        5. Stores results in `self.results` and plot paths in `self.plot_paths`
        
        The results are extracted from a markdown code block in the format:
        ```markdown
        [experiment results content]
        ```
        
        Args:
            data_description: Description of the data and tools available for the experiment.
            **kwargs: Additional keyword arguments passed to the underlying agent system.
        
        Raises:
            AgentExecutionError: If the experiment execution fails or results cannot be extracted
                from the agent's response.
            ContentExtractionError: If the results cannot be extracted from the markdown code block
                format.
        
        Note:
            After successful execution, `self.results` will contain the cleaned experiment results
            and `self.plot_paths` will contain a list of paths to generated plots/images.
        """

        logger.info(f"Starting experiment with configuration:")
        logger.info(f"  Engineer model: {self.engineer_model}")
        logger.info(f"  Researcher model: {self.researcher_model}")
        logger.info(f"  Planner model: {self.planner_model}")
        logger.info(f"  Plan reviewer model: {self.plan_reviewer_model}")
        logger.info(f"  Max n attempts: {self.max_n_attempts}")
        logger.info(f"  Max n steps: {self.max_n_steps}")
        logger.info(f"  Restart at step: {self.restart_at_step}")
        logger.info(f"  Hardware constraints: {self.hardware_constraints}")

        cmbagent = _import_cmbagent()
        results = cmbagent.planning_and_control_context_carryover(data_description,
                            n_plan_reviews = 1,
                            max_n_attempts = self.max_n_attempts,
                            max_plan_steps = self.max_n_steps,
                            max_rounds_control = 500,
                            engineer_model = self.engineer_model,
                            researcher_model = self.researcher_model,
                            planner_model = self.planner_model,
                            plan_reviewer_model = self.plan_reviewer_model,
                            plan_instructions=self.planner_append_instructions,
                            researcher_instructions=self.researcher_append_instructions,
                            engineer_instructions=self.engineer_append_instructions,
                            work_dir = self.experiment_dir,
                            api_keys = self.api_keys,
                            restart_at_step = self.restart_at_step,
                            hardware_constraints = self.hardware_constraints,
                            default_llm_model = self.orchestration_model,
                            default_formatter_model = self.formatter_model
                            )
        chat_history = results['chat_history']
        final_context = results['final_context']
        
        try:
            task_result = get_task_result(chat_history, 'researcher_response_formatter')
        except TaskResultError as e:
            logger.error(f"Failed to extract experiment results from chat history: {e}")
            raise AgentExecutionError(
                agent_name="researcher",
                stage="result_extraction",
                message=f"Could not extract experiment results from agent execution: {e}"
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
                    content_type="results",
                    source="researcher_response_formatter",
                    message=(
                        f"Could not extract results from markdown code block. "
                        f"Expected format: ```markdown ... ```. "
                        f"Task result length: {len(task_result)} characters."
                    )
                )
            extracted_results = matches[0]
        except IndexError as e:
            logger.error(f"Failed to extract results from markdown block: {e}")
            raise ContentExtractionError(
                content_type="results",
                source="researcher_response_formatter",
                message=f"Failed to extract results from formatted response: {e}"
            ) from e
        
        clean_results = re.sub(r'^<!--.*?-->\s*\n', '', extracted_results)
        self.results = clean_results
        self.plot_paths = final_context.get('displayed_images', [])

        return None



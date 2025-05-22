from langchain_core.runnables import RunnableConfig
from .parameters import GraphState
from .prompts import novelty_prompt, summary_literature_prompt
from ..paper_agents.tools import extract_latex_block, LLM_call_stream, json_parser3
import time
import requests
from tqdm import tqdm


    
# This node determines if an idea is novel or not. It may also ask for literature search
def novelty_decider(state: GraphState, config: RunnableConfig):
    """
    The goal of this function is to determine if an idea is novel or not. The function is given access to semantic scholar to find papers related to the project idea.
    """
    
    print(f"\nAddressing idea novelty: round {state['literature']['iteration']}")

    # check if idea is novel or not
    PROMPT = novelty_prompt(state)

    # Try for three times in case it fails 
    for _ in tqdm(range(5), desc="Analyzing novelty", unit="try"):
        
        state, result = LLM_call_stream(PROMPT, state)
        try:
            result    = json_parser3(result)
            reason    = result["Reason"]
            decision  = result["Decision"]
            query     = result["Query"]
            messages = f"{state['literature']['messages']}\nIteration {state['literature']['iteration']}\ndecision:{decision}\nreason:{reason}\n"
            iteration = state['literature']['iteration'] + 1
            break
        except Exception:
            time.sleep(2)

    else:
        raise Exception('Failed to extract json after 5 attempts')
        
    # get the reason for the decision
    if 'not novel' in decision.lower():
        print('Decision made: not novel')
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "literature_summary"}}
    
    elif 'novel' in decision.lower() or iteration>=state['literature']['max_iterations']:
        decision = 'novel'
        print('Decision made: novel')
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "literature_summary"}}

    else:
        # Get the value of the "Query" field
        print('Decision made: query')
        print(f'Query: {query}')
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "semantic_scholar"}}
    

# This node will search the arXiv for papers similar to the ones in the query
def semantic_scholar(state: GraphState, config: RunnableConfig):
    """
    This agent will search for papers given the search query and return the list of found ones
    """
    
    # search papers given the query
    results = SSAPI(state['literature']['query'], limit=10)

    total_papers = results.get("total", []) #total number of relevant papers found
    papers       = results.get("data",  []) #the actual data of the retrieved papers

    # A list with the idx, title, abstract, and url of the found papers. To be passed to the other agent
    papers_str = [] 
    if papers:
        print(f"Found {total_papers} potentially relevant papers")

        # do a loop over the papers
        for idx, paper in enumerate(papers, start=0):

            # get the year, abstract and url
            authors = ", ".join([author.get("name", "Unknown") for author in paper.get("authors", [])])
            title    = paper.get("title",    "No Title")
            year     = paper.get("year",     "Unknown Year")
            abstract = paper.get("abstract", "No Abstract")
            url      = paper.get("url",      "No URL")

            # string with paper information
            paper_str = f"""{idx+state['literature']['num_papers']}. {title} ({year})\nAuthors: {authors}\nAbstract: {abstract}\nURL: {url}\n\n"""
            
            # put these papers in the literature.log
            with open(f"{state['files']['literature_log']}", 'a', encoding='utf-8') as f:
                f.write(paper_str)

            # put papers into the papers_processed.log file
            with open(f"{state['files']['papers']}", 'a', encoding='utf-8') as f:
                f.write(paper_str)

            paper_str = papers_str.append(f"""{idx+state['literature']['num_papers']}. {title} \nAbstract: {abstract}\nURL: {url}\n""")
    else:
        papers_str.append("No papers found with the query.\n")

    total_papers_found = state['literature']['num_papers'] + min(len(papers), 10)
    print('Total papers analyzed', total_papers_found)
    
    return {"literature": {**state['literature'], 'papers': papers_str, "num_papers":total_papers_found}}


def SSAPI(query, limit=10) -> list:
    """
    Search for papers similar to the given query using Semantic Scholar API.

    Args:
        query (str): The search query (e.g., keywords or paper title).
        keys (dict): a python dictionary containing the session keys, including the semantic scholar one
        limit (int): Number of papers to retrieve (default is 10).

    Returns:
        list: A list of dictionaries containing paper details.
    """

    # Base URL for Semantic Scholar API
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {"query": query,
              "limit": limit,
              "fields": "title,authors,year,abstract,url"}

    # Conditionally include headers if API_KEY is available
    #if keys.SEMANTIC_SCHOLAR:
    #    response = requests.get(BASE_URL, headers={"x-api-key": keys.SEMANTIC_SCHOLAR},
    #                            params=params)
    #else:

    for _ in tqdm(range(200), desc="Calling Semantic Scholar", unit="try"):
        response = requests.get(BASE_URL, params=params)
        if response.status_code==200:
            return response.json()
        else:
            time.sleep(0.5) #wait for 1/2 second before retrying
        
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []


def literature_summary(state: GraphState, config: RunnableConfig):
    """
    This agent will take all messages from previous iterations and write a summary of the findings
    """

    # generate the summary
    PROMPT = summary_literature_prompt(state)    
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "SUMMARY")

    # write summary to file
    with open(f"{state['files']['literature']}", 'w', encoding='utf-8') as f:
        f.write(f"Idea {state['literature']['decision']}\n\n")
        f.write(text)

    # print out the summary
    print(text)

    print(f"done {state['tokens']['ti']} {state['tokens']['to']}")

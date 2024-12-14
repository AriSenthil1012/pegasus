import sys
import os

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Add the parent directory (pegasus) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import graphviz as gv
import re, subprocess

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.agents import tool
from langchain.agents.agent_toolkits import create_retriever_tool

from roscribe.prompts import get_gen_code_prompt, get_edit_code_prompt,\
    get_gen_launch_prompt, get_edit_launch_prompt, get_gen_package_prompt, get_edit_package_prompt,\
    get_gen_cmake_prompt, get_edit_cmake_prompt, get_gen_readme_prompt, get_edit_readme_prompt



def get_rag_tools(ros_distro):
    """
    Retrieves RAG tools for the specified ROS distribution.

    Args:
        ros_distro (str): The ROS distribution.

    Returns:
        rag_tool: The RAG tool for searching and returning documents regarding the ROS repositories.
    """

    # A database name is generated dynamically using the provided ROS distribution
    db_name = "ros_index_db_{}".format(ros_distro)

    # The function initializes a Chroma vector store with a persistence directory, where the database files for the specified ROS distribution are stored 
    # `Embeddings` provided by the OpenAIEmbeddings function for semantically similarity searches 
    vectorstore = Chroma(persist_directory="ROS_index_database/" + db_name, embedding_function=OpenAIEmbeddings()) 

    # The `vectorstore` is converted into a retriever, uses "mmr" - Maximal Marginal Relevance as the search strategy, optimizing for both diversity and relevance
    # While searching, the `k` parameter is set to 8, which limits the number of documents to be returned
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 8})


    # The function creates a RAG tool for searching and returning documents regarding the ROS repositories
    rag_tool = create_retriever_tool(
        retriever,
        "search_ROS_repositories",
        "Search and returns documents regarding the ROS repositories"
    )

    return rag_tool

def get_gen_graph_tool(agent):
    """
    Returns a function that shows the ROS graph and generates a description of the ROS graph in natural language.

    Parameters:
        agent: The agent object used to predict the ROS graph dictionary.

    Returns:
        show_ROS_graph: A function that shows the ROS graph and generates a description of the ROS graph.
    """

    # Define a tool called `show_ROS_graph` using the `@tool` decorator - the simplest way to create a custom tool 
    @tool 
    def show_ROS_graph() -> str:
        """
        Generates and displays a ROS graph based on the agent's predicted ROS graph dictionary.
        Returns a description of the ROS graph in natural language.
        """

        # Ask the agent for the predicted ROS graph as a dictionary 
        ros_graph_dict = agent.predict_ros_graph_dict()

        # Check if the dictionary is valid and non-empty 
        # If not, return an explanation that the graph can't be generated yet.
        if not isinstance(ros_graph_dict, dict) or len(ros_graph_dict.keys()) == 0:
            return "The graph if empty since there hasn't been enough conversation with the human user ."
        
        # Initialize a Graphviz Digraph object to create the ROS graph 
        graph = gv.Digraph()
        graph.attr(rankdir='LR') # Set the direction of the graph from left to right

        topic_list = [] # Keep track of topics already added to the group

        # Iterate over the nodes in the ROS graph dictionary
        for node in ros_graph_dict.keys():
            # Add each node to the graph as an ellipse-shaped node. 
            graph.node(node, style='solid', color='black', shape='ellipse')

            # Process the topics that the node subscribes to. 
            for sub_topic in ros_graph_dict[node]['subscribed_topics']:
                # Add the topic to the graph as a box if it hasn't been added yet.
                if sub_topic not in topic_list:
                    topic_list.append(sub_topic)
                    graph.node(sub_topic, stype='solid', color='black', shape='box')
                # Add an edge from the topic to the node.
                graph.edge(sub_topic, node)

            # Process the topics that the node publishes to.
            for pub_topic in ros_graph_dict[node]['published_topics']:
                # Add the topic to the graph as a box if it hasn't been added yet.
                if pub_topic not in topic_list:
                    topic_list.append(pub_topic)
                    graph.node(pub_topic, style='solid', color='black', shape='box')
                # Add an edge from the node to the topic.
                graph.edge(node, pub_topic)

        # Render and display the graph 
        graph.view()

        # Generate a natural language description of the ROS graph using the agent.
        gen_msg = "Here is a description of the show ROS graph:\n" + agent.get_ros_node_desc()

        # Return the generated description
        return gen_msg
    
    # Return the `show_ROS_graph` tool as the output of the 'get_gen_graph_tool' function.
    return show_ROS_graph

def get_code_retrieval_tool(agent):
    """
    Returns a code retrieval tool function that takes a checkout URI and VCS version of a GitHub repository
    and downloads the code.

    Parameters:
        agent (object): The agent object.

    Returns:
        function: The code retrieval tool function.
    """
    @tool
    def download_code(checkout_uri: str, vcs_version: str) -> str:
        """
        Takes a GitHub repository's checkout URI and a VCS (Version Control System) version
        (e.g., branch name or commit hash) and downloads the code into the agent's ROS workspace.

        Args:
            checkout_uri (str): The Git URI for the repository to be cloned.
            vcs_version (str): The branch, tag, or commit hash to clone.

        Returns:
            str: A message indicating whether the code was successfully downloaded or if there was an error.
        """

        # Define a regex pattern to extract the repository name from the Git URI.
        regex = r"\/(\w+)\.git"
        matches = re.findall(regex, checkout_uri)

        # Extract the repository name or return an error if the URI is invalid.
        try:
            repo_name = matches[0]  # Extract the first match from the regex (the repo name).
        except:
            # If the URI is invalid, return a helpful error message.
            return "Code download was unsuccessful due to incorrect Git URI.\n" \
                "Make sure the 'checkout_uri' is a Git URI.\n" \
                "Instead, set the search query to 'Repository summary for REPO_NAME' where REPO_NAME is the name of the repository."
        
        # Construct the Git clone command using the provided URI and version.
        git_command = "git clone {uri} -b {ver}".format(uri=checkout_uri, ver=vcs_version)

        # Log the repository name and the constructed Git command for debugging.
        print(f'Downloading code from {repo_name} using:\n{git_command}')

        try:
            # Attempt to execute the Git clone command in the agent's workspace directory.
            # The command changes the directory to `agent.ws_name/src` before running the clone command.
            subprocess.check_output(f'cd {agent.ws_name}/src && {git_command}', shell=True)
            
            # If successful, construct a success message.
            ret_msg = f'Code for {repo_name} successfully downloaded in the ROS workspace!'

            # Update the agent with node information, associating the repository with the workspace.
            node_info = {'code': 'RAG', 'readme': f'{repo_name}'}
            agent.update_node(node_info)

        except subprocess.CalledProcessError:
            # If the Git clone command fails, construct an error message.
            ret_msg = f'Code download for {repo_name} was unsuccessful!'

        # Print the final status message for debugging or user feedback.
        print(ret_msg)

        # Return the status message to the caller.
        return ret_msg

    # Return the `download_code` tool as the result of this function.
    return download_code

def get_code_gen_tool(agent):
    """
    Returns a function that generates code for a ROS node based on a given coding task.

    Parameters:
    - agent: The agent object that contains information about the ROS nodes and the language model.

    Returns:
    - write_ros_node: A function that takes a coding task as input and generates the code for a ROS node in Python.
    """
    @tool 
    def write_ros_node(coding_task: str) -> str:
        """
        Takes a very detailed coding task in natural language and implements the code in Python.

        Args:
            coding_task (str): The detailed coding task in natural language.

        Returns:
            str: A message indicating the success or failure of the code generation.
        """

        # Check if the current node exists in the agent's nodes dictionary
        # ROS nodes themselves do not have "keys". The `agent` is keeping track of ROS nodes in a dictionary like structure, 
        # (agent.nodes) where each node has a corresponding key that allows the agent to look up information about the node
        if agent.current_node not in agent.nodes.keys():
            # If there is no current node, use a default prompt for generating code
            gen_code_prompt = get_gen_code_prompt()

        else:
            # Check if the current node's code is marked as 'RAG' (not needing generation)
            if agent.nodes[agent.current_node]['code'] == 'RAG':
                # Return a message indicating that there is no need to generate code for this ROS node
                return "There is no need to generate code for this ROS node!"
            else:
                # Otherwise, get a prompt for editing the existing code of the node
                gen_code_prompt = get_edit_code_prompt(code=agent.nodes[agent.current_node]['code'])

            # Create a chain to generate the code, using the language model and prompt
            gen_code_chain = LLMChain(llm=agent.llm, prompt=gen_code_prompt, verbose=agent.verbose)
            
            # Use the chain to predict/generate code based on the given coding task
            gen_code_output = gen_code_chain.predict(task=coding_task)

            # Parse the generated code output to check if it's successful 
            parsed_output, successful = parse_code_gen(gen_code_output)

            if successful:
                # If code generation was successful, update the agent's node with the new code 
                agent.update_node(parsed_output)

                # Generate a message containing the implementation of the ROS node in Python
                gen_msg = "Python implementation of the ROS node:\n{code}\n\n{readme}".format(
                    code=parsed_output['code'],
                    readme=parsed_output['readme'])
                
                # Prepare the return message indicating success
                ret_msg = "Python implementation of the ROS node is successfully done!"
            else:
                # If code generation was unsuccessful, set messages indicating failure
                gen_msg = "Python implementation of the ROS node was unsuccessfull"
                ret_msg = gen_msg
            
            # Print the detailed message with code or failure information
            print(gen_msg)
            # Return the summary message
            return ret_msg
    return write_ros_node

def get_launch_tool(agent):
    """
    This function returns a tool function that can be used to edit a ROS launch file based on the given input.

    Parameters:
        agent: The agent object.

    Returns:
        The tool function that takes a natural language task as input and returns a string.
    """
    @tool
    def edit_launch_file(launch_file_edit_task: str) -> str:
        """This function takes a natural language task `(launch_file_edit_task)` as input and returns a string. 
           It aims to edit a ROS lauch file based on the given input 
           
           Args:
            coding_task (str): The detailed coding task in natural language.

           Returns:
            str: A message indicating the success or failure of the code generation."""

        # The code checks if there is an existing launch file in the agent's package.
        # It uses the agent's project name to dynamically construct the key for the launch file.
        if agent.package[f'{agent.project_name}.launch'] is None:
            # If the launch file does not exist (is None), then generate a prompt to create a new launch file.
            gen_launch_prompt = get_gen_launch_prompt(agent=agent)
            # Set edit flag to False because we are generating a new file, not editing.
            edit = False 
        else:
            # If the launch file exists, generate a prompt to edit the existing launch file.
            gen_launch_prompt = get_edit_launch_prompt(launch_file=agent.package[f'{agent.project_name}.launch'])
            # Set edit flag to True since we are editing an existing launch file.
            edit = True

        # Create an LLMChain object to generate the required content based on the prompt.
        # The LLMChain takes in the agent's language model, the generated prompt, and the verbosity setting.
        gen_launch_chain = LLMChain(llm=agent.llm, prompt=gen_launch_prompt, verbose=agent.verbose)

        if edit:
            # If we are editing the file, call the LLMChain's predict method with the task to generate the edits.
            gen_launch_output = gen_launch_chain.predict(task=launch_file_edit_task)
        else:
            # If we are generating a new file, call the predict method without additional task input.
            # Note: There is a typo here. It should be `=` instead of `-` to assign the generated output.
            gen_launch_output = gen_launch_chain.predict()

        # Parse the generated code to extract useful content and check if it was successful.
        parsed_output, successful = parse_code_gen(gen_launch_output)

        if successful:
            # If the code parsing was successful, update the agent's package with the generated launch file content.
            agent.package[f'{agent.project_name}.launch'] = parsed_output['code']

            # Prepare a message indicating that the ROS launch file has been generated.
            gen_msg = "Generated ROS launch file for the ROS package:\n{launch_file}".format(launch_file=parsed_output['code'])

            # Prepare a return message indicating that the file has been successfully edited.
            ret_msg = "The ROS launch file for the ROS package has been successfully edited!"
        else:
            # If the parsing was not successful, indicate that the modification of the launch file was unsuccessful.
            gen_msg = "Modification of the ROS launch file for the ROS package was unsuccessful!"
            ret_msg = gen_msg

        # If we were editing the launch file (not generating a new one), print the generated message.
        if edit:
            print(gen_msg)

        # Return the appropriate message indicating the outcome of the edit or generation process.
        return ret_msg

        # The function returns `edit_launch_file` so it can be used as a callable tool elsewhere.
    return edit_launch_file

def get_package_tool(agent):
    """
    Returns a tool function that takes feedback in natural language from the agent in order to edit a ROS package.xml file.

    Parameters:
        agent: The agent object.

    Returns:
        The tool function that edits the package.xml file.
    """
    @tool
    def edit_package_xml(ros_package_edit_task: str) -> str:
        """
        Takes feedback in natural language from the agent in order to edit a ROS package.xml file.

        Args:
            ros_package_edit_task (str): The feedback in natural language describing the edit task.

        Returns:
            str: A message indicating the success or failure of the package.xml file edit.
        """

        # Check if the package.xml file exists in the agent's package dictionary
        if agent.package['package.xml'] is None:
            # If it doesn't exist, get the prompt for generating a new package.xml
            gen_package_prompt = get_gen_package_prompt(agent=agent)
            edit = False
        else:
            # If it exists, get the prompt for editing the existing package.xml
            gen_package_prompt = get_edit_package_prompt(package=agent.package['package.xml'])
            edit = True

        # Create an LLMChain object with the generated/edit prompt
        gen_package_chain = LLMChain(llm=agent.llm, prompt=gen_package_prompt, verbose=agent.verbose)

        if edit:
            # If editing an existing package.xml, predict the output based on the edit task
            gen_package_output = gen_package_chain.predict(task=ros_package_edit_task)
        else:
            # If generating a new package.xml, predict the output without a specific task
            gen_package_output = gen_package_chain.predict()

        # Parse the generated/edit output and check if it was successful
        parsed_output, successful = parse_code_gen(gen_package_output)

        if successful:
            # If successful, update the agent's package dictionary with the new/edited package.xml
            agent.package['package.xml'] = parsed_output['code']

            gen_msg = "Generated package.xml for the ROS package:\n{package}".format(package=parsed_output['code'])

            ret_msg = "The package.xml file for the ROS package has been successfully edited!"
        else:
            gen_msg = "Modification of the package.xml file for the ROS package was unsuccessful!"
            ret_msg = gen_msg

        if edit:
            # If editing an existing package.xml, print the generated message
            print(gen_msg)

        return ret_msg
    return edit_package_xml

def get_cmake_tool(agent):
    """
    Returns a function that takes feedback in natural language from the agent in order to edit a CMakeLists.txt file for a ROS package.

    Parameters:
    - agent: The agent object.

    Returns:
    - edit_cmake: The function that edits the CMakeLists.txt file.
    """
    @tool
    def edit_cmake(ros_cmake_edit_task: str) -> str:
        """Takes feedback in natural language from the agent in order to edit a CMakeLists.txt file for a ROS package.

        Args:
            ros_cmake_edit_task (str): The feedback in natural language from the agent.

        Returns:
            str: A message indicating the success or failure of the CMakeLists.txt file editing.
        """

        # Check if CMakeLists.txt file exists in the package
        if agent.package['CMakeLists.txt'] is None:
            gen_cmake_prompt = get_gen_cmake_prompt(agent)
            edit = False
        else:
            gen_cmake_prompt = get_edit_cmake_prompt(cmake=agent.package['CMakeLists.txt'])
            edit = True

        # Create LLMChain for generating/editing CMakeLists.txt
        gen_cmake_chain = LLMChain(llm=agent.llm, prompt=gen_cmake_prompt, verbose=agent.verbose)

        # Generate or edit CMakeLists.txt based on the task
        if edit:
            gen_cmake_output = gen_cmake_chain.predict(task=ros_cmake_edit_task)
        else:
            gen_cmake_output = gen_cmake_chain.predict()

        # Parse the generated/edit output and check if successful
        parsed_output, successful = parse_code_gen(gen_cmake_output)

        # Update the CMakeLists.txt in the package if successful
        if successful:
            agent.package['CMakeLists.txt'] = parsed_output['code']

            gen_msg = "Generated CMakeLists.txt for the ROS package:\n{cmake}".format(cmake=parsed_output['code'])

            ret_msg = "The CMakeLists.txt file for the ROS package has been successfully edited!"
        else:
            gen_msg = "Modification of the CMakeLists.txt file for the ROS package was unsuccessful!"
            ret_msg = gen_msg

        # Print the generated message if editing
        if edit:
            print(gen_msg)

        return ret_msg

    return edit_cmake

def get_readme_tool(agent):
    """
    Returns a tool function that can be used to edit a README.md file for a ROS package.

    Parameters:
        agent: The agent object.

    Returns:
        The tool function that can be used to edit the README.md file.
    """
    @tool
    def edit_readme(ros_readme_edit_task: str) -> str:
        """
        Takes feedback in natural language from the agent in order to edit a README.md file for a ROS package.

        Args:
            ros_readme_edit_task (str): The feedback in natural language from the agent.

        Returns:
            str: A message indicating the success or failure of editing the README.md file.
        """
        # Check if README.md exists in the package
        if agent.package['README.md'] is None:
            gen_readme_prompt = get_gen_readme_prompt(agent)
            edit = False
        else:
            gen_readme_prompt = get_edit_readme_prompt(readme=agent.package['README.md'])
            edit = True

        # Create LLMChain for generating README.md
        gen_readme_chain = LLMChain(llm=agent.llm, prompt=gen_readme_prompt, verbose=agent.verbose)

        # Generate or edit README.md based on the task
        if edit:
            gen_readme_output = gen_readme_chain.predict(task=ros_readme_edit_task)
        else:
            gen_readme_output = gen_readme_chain.predict()

        # Parse the generated output and check if successful
        parsed_output, successful = parse_code_gen(gen_readme_output)

        if successful:
            agent.package['README.md'] = parsed_output['code']

            gen_msg = "Generated README.md for the ROS package:\n{readme}".format(readme=parsed_output['code'])

            ret_msg = "The README.md file for the ROS package has been successfully edited!"
        else:
            gen_msg = "Modification of the README.md file for the ROS package was unsuccessful!"
            ret_msg = gen_msg

        # Print the generated message if editing
        if edit:
            print(gen_msg)

        return ret_msg

    return edit_readme


def get_file_tool(agent):
    """
    Returns a tool function that can be used to load the content of a file.

    Parameters:
        agent: The agent object.

    Returns:
        The load_file function.

    """

    @tool
    def load_file(file_name: str) -> str:
        """Takes a file name and loads the content of the file."""

        file_address = f"{agent.ws_name}/src/{agent.project_name}/"

        if '.py' in file_name:
            file_name = file_address + "src/" + file_name
        elif '.launch' in file_name:
            file_name = file_address + "launch/" + file_name
        else:
            file_name = file_address + file_name

        try:
            f = open(file_name, "r")
            content = f.read()
            return content
        except OSError:
            return "File cannot be opened! It can be due to an incorrect file name, or the file might be missing!"

    return load_file

def parse_code_gen(code_gen_output):
    """
    Parses the code_gen_output and extracts the code and readme.

    Args:
        code_gen_output (str): The output generated by the code generator.

    Returns:
        tuple: A tuple containing the parsed code and readme.
            The code is a string containing the extracted code.
            The readme is a string containing the extracted readme.

    Raises:
        None
    """
    # Define regex pattern to match code block
    regex = r"```[^\n]*\n(.+?)```"
    # Attempt to match the regex pattern with the code_gen_output
    match = re.match(regex, code_gen_output, re.DOTALL)

    if match is None:
        # Return error message if code cannot be parsed
        return {'code': 'Generated code cannot be parsed!', 'readme': 'Generated README cannot be parsed!'}, False

    # Extract code and readme from the code_gen_output
    code = match.group(1)
    readme = code_gen_output.split("```")[-1]

    # Return parsed code and readme
    return {'code': code, 'readme': readme}, True
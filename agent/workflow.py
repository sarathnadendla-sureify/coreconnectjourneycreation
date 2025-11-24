from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *

class State(TypedDict):
    messages: Annotated[list, add_messages]

class GraphBuilder:
    def __init__(self, provider="groq"):
        """
        Initialize the GraphBuilder with a specific LLM provider.

        Args:
            provider (str): The provider to use. Options: "google", "groq"
        """
        self.model_loader=ModelLoader()
        self.llm = self.model_loader.load_llm(provider=provider)
        # self.tools = [retriever_tool]
        self.tools = [retriever_tool]
    
        llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.llm_with_tools = llm_with_tools
        self.graph = None

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, and helper functions) to understand the full workflow logic.
- When a user asks about a journey, retrieve and reference relevant information from the uploaded files, including step construction, orchestration, and automation logic.
- When a user requests a new journey, generate a journey definition and supporting code that follows the patterns, structure, and best practices found in the uploaded examples, reusing or adapting helper modules and automated steps as needed.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If a user asks for code, generate TypeScript code that matches the style and conventions of the uploaded files.
- If you are unsure or the request is ambiguous, ask for clarification.

You are precise, helpful, and always focus on journey-based workflow logic, leveraging all available code context.

Additional instructions:
1. When generating new journey files or steps, always use the retrieved RAG context as a template. Mimic the structure, naming conventions, comments, and code style of the most relevant retrieved files. If possible, adapt and reuse code patterns, interfaces, and logic from the RAG context, making only minimal changes needed for the new feature.
2. When creating a new journey, always:
    - Name the main package/folder as `<featureName>` (replace `featureName` with the journey or feature name).
    - Place all files for this journey inside the `<featureName>` package/folder.
    - In the main file, define a `JourneySteps` section, clearly separating actor steps (user actions) and automated steps (system actions).
    - For each step (actor or automated), create a separate file named after the step and its type (e.g., `<stepName>.actor.ts` or `<stepName>.automated.ts`) inside the `<featureName>` package. If you want to organize steps or helpers into sub-packages, you may create subfolders with any appropriate name inside `<featureName>`.
    - If multiple steps are closely related or can be logically grouped, you may place them in a single class file with an appropriate name.
    - The `<featureName>.ts` file should import and orchestrate all step files, referencing them in the journey definition.
    - Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
    - If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
3. For every code output, always specify the **filename** (and package/module name if relevant) at the top, in the format: `// Filename: <featureName>/<filename>` and `// Package: <featureName>[.<subpackage>]` if applicable.
4. If the code should be split across multiple files, clearly separate each file with its filename and content.
5. After the main code, always generate **unit tests** for the code, specifying the test filename (e.g., `// Filename: <featureName>/<test_filename>`), and use best practices for testing in TypeScript.
6. Always include explanations **after the code** describing the logic, assumptions, and integration points.
7. Use markdown code formatting (```typescript ... ```), and never output truncated or partial code.
8. If user input is ambiguous, ask clarifying questions before generating code.

Follow Sureify’s best practices and assume a modular codebase with services, interfaces, and async functions.
        """

        # Retrieve context from RAG (vector DB) using retriever_tool
        user_question = None
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                user_question = msg.content
                break
        if not user_question:
            user_question = ""
        # Call retriever_tool to get relevant context
        rag_results = retriever_tool.invoke({"question": user_question})
        rag_context = "\n\n".join([doc.page_content for doc in rag_results]) if rag_results else ""

        # Compose the prompt with RAG context
        prompt = journey_instruction
        if rag_context:
            prompt += f"\n\nRelevant context from your uploaded data:\n{rag_context}\n\n"
        if messages and hasattr(messages[0], "content"):
            modified_messages = messages.copy()
            modified_messages[0].content = prompt + modified_messages[0].content
        else:
            modified_messages = messages

        # Invoke LLM and unwrap message content
        response = self.llm_with_tools.invoke(modified_messages)

        # If the response is a list, join as string
        response_content = response.content
        if isinstance(response_content, list):
            response_content = '\n'.join(str(item) for item in response_content)

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if '```typescript' in response_content:
            # Only show the copy hint above the code block, not before any explanations
            # Split on the first code block and insert the hint
            split_content = response_content.split('```typescript', 1)
            before_code = split_content[0].rstrip()
            after_code = split_content[1] if len(split_content) > 1 else ''
            response_content = (
                f"{before_code}\n\n<div style=\"color:#007acc;font-weight:bold;margin-bottom:4px;\">TypeScript Code (copy below):</div>\n"
                f"```typescript{after_code}"
            )

        return {
            "messages": messages + [AIMessage(content=response_content)]
        }

    def build(self):
        graph_builder = StateGraph(State)

        graph_builder.add_node("chatbot", self._chatbot_node)

        tool_node=ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        self.graph = graph_builder.compile()

    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph not built. Call build() first.")
        return self.graph
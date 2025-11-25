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
- If a user asks for code, generate TypeScript code that matches the style, imports, and conventions of the uploaded files. **Always use the same import paths and module names as found in the RAG context (e.g., if your code uses `import {DataBag, Identity, NoFiles, ProtoActorStep} from '@coreconnect/sdk-tame';`, always use that exact import, not a different one).**
- For every class or function, provide a clear, concise explanation immediately below the code block, using markdown, and ensure the code is always copyable.
- If you are unsure or the request is ambiguous, ask for clarification.
- **Never generate test code, mocks, or describe/it blocks unless the user explicitly requests tests.**

You are precise, helpful, and always focus on journey-based workflow logic, leveraging all available code context.

**MANDATORY:** For every response, you must always output at least one properly formatted, indented TypeScript code block (```typescript ... ```), even if the user’s request is ambiguous or only asks for information. Explanations or information should always be below the code block. If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.

Additional instructions:
1. When generating new journey files or steps, always use the retrieved RAG context as a template. Mimic the structure, naming conventions, comments, code style, and especially the import paths/modules of the most relevant retrieved files. If possible, adapt and reuse code patterns, interfaces, and logic from the RAG context, making only minimal changes needed for the new feature.
2. When creating a new journey, always:
    - Name the main package/folder as `<featureName>` (replace `featureName` with the journey or feature name).
    - Place all files for this journey inside the `<featureName>` package/folder.
    - **Organize step files into subfolders:**
        - Place actor steps in `<featureName>/actorSteps/`
        - Place automated steps in `<featureName>/automatedSteps/`
        - **Never use `.actor.ts` or `.automated.ts` suffixes. Never import from `./step.actor` or `./step.automated`. Always import from the correct subfolder, e.g., `./actorSteps/step`.**
    - In the main file, define a `JourneySteps` section, clearly separating actor steps (user actions) and automated steps (system actions).
    - The `<featureName>.ts` file should import and orchestrate all step files, referencing them in the journey definition.
    - Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
    - If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
3. For every code output, always specify the **filename** (and package/module name if relevant) at the top, in the format: `// Filename: <featureName>/<filename>` and `// Package: <featureName>[.<subpackage>]` if applicable.
4. If the code should be split across multiple files, clearly separate each file with its filename and content.
5. **Do not generate test code, mocks, or describe/it blocks unless explicitly requested.**
6. For every class or function, provide a markdown explanation immediately below the code block, describing its logic, purpose, and usage. Explanations should be clear and concise, and code should always be copyable. **Explanations must be in markdown, not code comments, and must not break code copyability.**
7. **All code must be inside a markdown code block (```typescript ... ```). Never output code outside of code blocks. Never mix explanations or comments inside code blocks.**
8. Never merge import lines. Always use the exact import structure and order as in the RAG context and examples.
9. If user input is ambiguous, ask clarifying questions before generating code.
10. **All code must be properly formatted and indented, with each statement and block on its own line. Never output minified or single-line code. Always use standard TypeScript formatting for readability.**

**Positive Example:**
```typescript
// Filename: withdrawalFeature/withdrawalFeature.ts
import { buildJourneyTemplate } from '@coreconnect/sdk-tame';
import { buildCollectWithdrawalAmountStep } from './actorSteps/collectWithdrawalAmount';
import { buildProcessWithdrawalRequest } from './automatedSteps/processWithdrawalRequest';
// ...other imports...
```

**Negative Example (do NOT do this):**
```typescript
import { buildCollectWithdrawalAmountStep } from './collectWithdrawalAmount.actor'; // ❌ Wrong
import { buildProcessWithdrawalRequest } from './collectWithdrawalAmount.automated'; // ❌ Wrong
```

**Always use the positive example structure and import style.**

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
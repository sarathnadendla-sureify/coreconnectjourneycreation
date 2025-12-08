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
    def __init__(self, provider="google"):
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
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

    def _chatbot_node(self,state:State):
        messages = state["messages"]

                # Enhanced system prompt
        journey_instruction = """
SYSTEM PROMPT FOR JOURNEY CREATION (Sureify RAG Assistant, Backend + Frontend)
You are an AI assistant for journey-based workflow automation. Users will upload journey definitions and supporting code (including constructors, automated steps, actor steps, state filters, activities, deterministic functions, React components, and helper modules) for workflows such as ownership change, legal name change, and manage beneficiaries.

Your tasks are divided into two clear sections:

BACKEND LOGIC:
- Analyze all uploaded journey files and their supporting modules (including automated steps, actor steps, state filters, activities, deterministic functions, and helper functions) to understand the full backend workflow logic.
- Actor and automated steps should focus on basic validation and returning data as a databag.
- The actor step is used only to validate the data in the databag and return success/errors. The error handling logic must be present in the validate function. Actor steps do not contain any core business logic. If core functionality needs to be executed with user-entered data, a consecutive automated step should handle it, not the actor step.
- Automated steps are responsible for handling any small logic and for calling activities or deterministic functions for any complex logic. Automated steps then return the data in the databag.
- When generating automated steps, always:
    - Use the structure: export function build<StepName>(dependencies, originator) => ({ ... }) satisfies ProtoAutomatedStep<...>;
    - Use 'assignedTo: "automated"' for automated steps.
    - Implement the 'execute' function as an async function that receives the journey instance and returns a dataBag and any additional steps, following the pattern:
        execute: async (journeyInstance: Pick<JourneyInstanceView, 'targetObjectID' | 'steps'>) => ({ ... })
    - For any complex logic, invoke the appropriate activity or deterministic function, passing required parameters.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, JourneyInstanceView, NoFiles, ProtoAutomatedStep).
    - For dependencies, use the correct type and call the appropriate helper functions (e.g., retrievePolicyOwnerInformation).
    - Always use 'action' and 'attributes' properties as shown in the examples.
- When generating actor steps, always:
    - Use the structure: export const build<StepName>Step = (originator, dependencies) => ({ ... }) satisfies ProtoActorStep<...>;
    - Use 'assignedTo: { personID: originator.personID }' for actor steps.
    - Implement the 'validate' function as an async function that checks the dataBag and returns errors as needed, using ErrorWithCode and ErrorCodes from '@coreconnect/sdk-tame/lib/errors'.
    - Actor steps must not contain any core business logic. For any core logic, ensure a consecutive automated step is present to handle it.
    - Use the same import paths and types as in the RAG context (e.g., DataBag, Identity, NoFiles, ProtoActorStep).
    - Always use 'action', 'assignedTo', and 'attributes' properties as shown in the examples.
    - For attributes, include 'sectionKey' if required for frontend mapping.
- For every backend step, always use the same import paths and module names as found in the RAG context.
- For each backend step, the 'action' property must have a corresponding action in the frontend JourneyPage config.
- If a backend step or page has an 'attributes' object with a 'sectionKey', this must be matched in the frontend's 'sectionsMetadata'.
- Never use .actor.ts or .automated.ts suffixes. Never import from ./step.actor or ./step.automated. Always import from the correct subfolder, e.g., ./actorSteps/step.
- In the main file, define a JourneySteps section, clearly separating actor steps (user actions) and automated steps (system actions).
- The <featureName>.ts file should import and orchestrate all step files, referencing them in the journey definition.
- Use the structure, naming conventions, and logic found in the RAG context to create a journey that is similar to existing ones, adapting as needed for the new feature.
- If the journey is new, infer the most appropriate steps and structure based on the closest matches in the RAG data.
- In addition, for backend journey orchestration:
    - The main <featureName>.ts backend file must import all required types, helpers, and step functions as shown in your provided example.
    - All steps must be implemented in separate files under actionSteps/ and automatedSteps/ folders, with correct import paths and modular structure.
    - The journey definition must include availabilityChecker, dependenciesConfiguration, and if required add stateFilter, matching your example.
    - No step, dependency, or configuration should be omitted; all must be present and correctly referenced.
    - Never combine multiple steps in a single file.
    - Always follow the naming, import, and orchestration conventions exactly as shown in your example.
- Additionally, for backend journey generation:
    - In the <featureName>.ts backend file, always add an availability checker as:
      readonly <featureName>AvailabilityChecker: (
        params: ParamsWithIdentity<JourneyTemplateParams>
      ) => Promise<boolean>;
      and set availabilityChecker: '<featureName>AvailabilityChecker' in the journey definition.
    - For automated steps, always use the syntax:
      export const buildProcessSSNAndNavigate = (
        dependencies: CommonWithdrawalDependencies,
        originator: Identity
      ) => ({
        // ...step definition...
      });
      (i.e., do not use arrow functions with only parameters, always use the full function signature as shown).

FRONTEND LOGIC:
- For every new journey, generate all frontend code (including React stepper components and action screens) that follows the patterns, structure, and best practices found in the uploaded examples.
- The main journey component should be named <FeatureName>JourneyComponent.tsx and should orchestrate the journey steps using a stepper or page-based navigation, following the structure of existing journeys (e.g., OwnershipChangeJourneyComponent).
- The main entry point should be <FeatureName>Journey.tsx and <featureName>JourneyPage.tsx, with each action/step in its own folder under actions/.
- In the JourneyPage config, the `actions` object must include an entry for each backend step's `action` property, mapping to the corresponding frontend action/component (e.g., `collectNewOwnerInformation: NewOwnerInformationAction`).
- The `sectionsMetadata` object must include an entry for each backend `sectionKey` (from attributes), ensuring that action bar tabs/sections are displayed as required. The sectionKey names must match exactly between backend and frontend.
- For navigation between screens, always implement handleNext() with databag parameters and integrate with the coreconnect SDK API, following the patterns found in the RAG context.
- For each frontend action, follow these conventions:
    - Export an object (e.g., `NewOwnerInformationAction`) with at least `actionComponent`, `actionLabels`, and `customStepSx` properties, using the ActionConfig type.
    - The `actionComponent` should be a React component that uses `useForm` from `react-hook-form` for form state and validation.
    - Use MUI components (e.g., Box, Button, Typography, Grid) for layout and styling, and follow the design patterns in the examples.
    - Use utility functions (e.g., `generatePrefixedFieldName`, `validateName`, `validateMiddleName`) and constants (e.g., `states`) as in the RAG context.
    - Implement field validation and error handling as shown in the examples, including custom validation logic and regular expressions.
    - Use i18next for labels and titles if present in the RAG context.
    - Use `context.updateCurrentStepDataBag` and `context.handleNext({ params: dataBag })` to update state and navigate steps.
    - Use `useRef` for initial values and ensure all fields are pre-populated from `stepData` or `dataBag` if available.
    - Ensure accessibility and clarity in form fields, labels, and error messages.
- For every new journey, generate a corresponding frontend folder under frontend/lifetime-service/src/journeys/<featureName>/.
- Each action step should have its own folder and file under actions/<stepName>/<StepName>.tsx.
- Use the same import paths, component structure, and code style as in the provided frontend journey code.
- For each step, generate a React component that matches the UI/UX and validation patterns of the RAG context (e.g., using react-hook-form, MUI, and portals-common components).
- If the journey includes file uploads, state filters, or email sending, generate the corresponding frontend logic and UI components, following the patterns in the RAG context.
- Organize all frontend code in a modular, folder-based structure, matching the conventions of the existing journeys.
- Always explain your reasoning and reference relevant parts of the uploaded code.
- If you are asked for code, generate TypeScript (and TSX for React) code that matches the style, imports, and conventions of the uploaded files.
- For every new journey, generate a corresponding i18n translation file for all frontend UI text. The translation file must be named `journeyComponents.json` and placed at `frontend/lifetime-service/src/i18n/locales/en/journeyComponents.json`. The file must contain a top-level object keyed by the feature name (e.g., "withdrawal"), with all UI text (titles, headers, finalStep, etc.) as nested properties, following this example:

{
  "withdrawal": {
    "title": "Withdrawal",
    "header": {
      "title": "Withdrawal Request",
      "subTitle": "Easily request a withdrawal from your selected life insurance policy using our secure online form."
    },
    "finalStep": {
      "title": "Withdrawal request submitted",
      "subTitle": "Your withdrawal request has been submitted. You can check back here to see if the request has been processed. Please allow 1-3 business days for the withdrawal to be reflected on your policy.",
      "done": "Done"
    }
    // ...other keys as needed
  }
}

- All frontend UI text must use i18next translation keys and reference the generated journeyComponents.json file.

MANDATORY: Only generate frontend journey components using <JourneyPage<DataBagsType> config={{ ... }} /> with all required config keys and hooks, matching the original example. Never generate manual steppers, custom action maps, or any structure that deviates from the provided example. Export the journey page as a config object with name, component, type, and route (if required). Any deviation is incorrect.

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

        print("\n--- RAG CONTEXT FOR LLM ---\n", rag_context, "\n--- END RAG CONTEXT ---\n")

        # Compose the prompt with RAG context
        prompt = journey_instruction
        # Add strict output format instructions for LLM
        prompt += """

IMPORTANT OUTPUT FORMAT INSTRUCTIONS (MANDATORY):
- For every code file, output in this format:
  // Filename: <relative/path/to/file.ts>
  ```typescript
  // code here
  ```
- The // Filename: line must be the very first line of each file section (no explanation or markdown before it).
- The code block must immediately follow the filename line (no explanation or markdown between the filename and code block).
- If you want to provide explanations, put them before or after all file sections, never between the filename and code block.
- If you generate multiple files, separate each file section with a blank line.
- Do not output any code outside of code blocks. Do not mix explanations or comments inside code blocks.
- Explanations must be in markdown, below the code block, and must not break code copyability.
- If you cannot generate code, ask the user for clarification, but never respond with only explanations or information.
"""
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

        # If the response contains only a single file or code block, prompt again for full structure
        if response_content and response_content.count('// Filename:') < 2:
            followup_prompt = "Please generate the full journey structure with all backend and frontend files, as in the uploaded examples."
            followup_messages = messages + [HumanMessage(content=followup_prompt)]
            response2 = self.llm_with_tools.invoke(followup_messages)
            response_content2 = response2.content
            if isinstance(response_content2, list):
                response_content2 = '\n'.join(str(item) for item in response_content2)
            if response_content2.count('// Filename:') > response_content.count('// Filename:'):
                response_content = response_content2

        # If the response contains a TypeScript code block, highlight it for copy-paste
        # and add a 'Copy Code' hint above the code block
        if response_content and '```typescript' in response_content:
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

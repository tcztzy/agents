import os
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

llm_config = {"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]}


assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    },
)
assistant.reset()
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem="What is autogen?")

import os
import asyncio
import warnings
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.ERROR)
print(
    f"Google API Key set: {'Yes' if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_API_KEY') != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}"
)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or "0"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "False"


# --- Tool Definitions ---

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city."""
    print(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "")

    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]

    return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting."""
    if name:
        print(f"--- Tool: say_hello called with name: {name} ---")
        return f"Hello, {name}!"

    print("--- Tool: say_hello called without a specific name ---")
    return "Hello there!"


def say_goodbye() -> str:
    """Provides a simple farewell message."""
    print("--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."


def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather and formats temperature based on session state."""
    print(f"--- Tool: get_weather_stateful called for {city} ---")
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized not in mock_weather_db:
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}

    data = mock_weather_db[city_normalized]
    temp_c = data["temp_c"]
    condition = data["condition"]

    if preferred_unit == "Fahrenheit":
        temp_value = (temp_c * 9 / 5) + 32
        temp_unit = "°F"
    else:
        temp_value = temp_c
        temp_unit = "°C"

    report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
    result = {"status": "success", "report": report}
    print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

    tool_context.state["last_city_checked_stateful"] = city
    print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

    return result


def block_keyword_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Blocks model execution when the latest user message contains BLOCK."""
    agent_name = callback_context.agent_name
    print(f"--- Callback: block_keyword_guardrail running for agent: {agent_name} ---")

    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break

    print(f"--- Callback: Inspecting last user message: '{last_user_message_text[:100]}...' ---")
    keyword_to_block = "BLOCK"

    if keyword_to_block in last_user_message_text.upper():
        print(f"--- Callback: Found '{keyword_to_block}'. Blocking LLM call! ---")
        callback_context.state["guardrail_block_keyword_triggered"] = True
        print(f"--- Callback: Set state 'guardrail_block_keyword_triggered': True ---")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
        )

    print(f"--- Callback: Keyword not found. Allowing LLM call for {agent_name}. ---")
    return None


def block_paris_tool_guardrail(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
    """Blocks the get_weather_stateful tool when the city argument is Paris."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    print(f"--- Callback: block_paris_tool_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")

    target_tool_name = "get_weather_stateful"
    blocked_city = "paris"

    if tool_name == target_tool_name:
        city_argument = args.get("city", "")
        if city_argument and city_argument.lower() == blocked_city:
            print(f"--- Callback: Detected blocked city '{city_argument}'. Blocking tool execution! ---")
            tool_context.state["guardrail_tool_block_triggered"] = True
            print(f"--- Callback: Set state 'guardrail_tool_block_triggered': True ---")
            return {
                "status": "error",
                "error_message": f"Policy restriction: Weather checks for '{city_argument.capitalize()}' are currently disabled by a tool guardrail.",
            }

    print(f"--- Callback: Allowing tool '{tool_name}' to proceed. ---")
    return None


def make_sub_agents() -> tuple[Agent, Agent]:
    """Creates fresh greeting and farewell sub-agents for each root agent."""
    greeting_agent = Agent(
        model="gemini-2.5-flash",
        name="greeting_agent",
        instruction=(
            "You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
            "Use the 'say_hello' tool to generate the greeting. If the user provides their name, make sure to pass it to the tool. "
            "Do not engage in any other conversation or tasks."
        ),
        description="Handles simple greetings and hellos using the 'say_hello' tool.",
        tools=[say_hello],
    )
    farewell_agent = Agent(
        model="gemini-2.5-flash",
        name="farewell_agent",
        instruction=(
            "You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
            "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation. "
            "Do not perform any other actions."
        ),
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
        tools=[say_goodbye],
    )
    return greeting_agent, farewell_agent


def print_response(label: str, text: str) -> None:
    print(f"<<< {label}: {text}")


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> None:
    print(f"\n>>> User Query: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"

    print_response("Agent Response", final_response_text)


async def main() -> None:
    # --- Base Weather Agent ---
    session_service = InMemorySessionService()
    APP_NAME = "weather_tutorial_app"
    USER_ID = "user_1"
    SESSION_ID = "session_001"

    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    weather_agent = Agent(
        name="weather_agent_v1",
        model="gemini-2.5-flash",
        description="Provides weather information for specific cities.",
        instruction=(
            "You are a helpful weather assistant. When the user asks for the weather in a specific city, "
            "use the 'get_weather' tool to find the information. If the tool returns an error, inform the user politely. "
            "If the tool is successful, present the weather report clearly."
        ),
        tools=[get_weather],
    )
    print(f"Agent '{weather_agent.name}' created using model '{weather_agent.model}'.")

    runner = Runner(agent=weather_agent, app_name=APP_NAME, session_service=session_service)
    print(f"Runner created for agent '{runner.agent.name}'.")

    await call_agent_async("What is the weather like in London?", runner, USER_ID, SESSION_ID)
    await call_agent_async("How about Paris?", runner, USER_ID, SESSION_ID)
    await call_agent_async("Tell me the weather in New York", runner, USER_ID, SESSION_ID)

    # --- Agent Team with Sub-Agents ---
    greeting_agent, farewell_agent = make_sub_agents()
    print(f"✅ Sub-agents created: {[greeting_agent.name, farewell_agent.name]}")

    weather_agent_team = Agent(
        name="weather_agent_v2",
        model="gemini-2.5-flash",
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
        instruction=(
            "You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
            "Use the 'get_weather' tool ONLY for specific weather requests. "
            "You have specialized sub-agents: 'greeting_agent' for greetings and 'farewell_agent' for farewells. "
            "If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
            "If it's a weather request, handle it yourself using 'get_weather'."
        ),
        tools=[get_weather],
        sub_agents=[greeting_agent, farewell_agent],
    )
    print(f"✅ Root Agent '{weather_agent_team.name}' created with sub-agents.")

    session_service_team = InMemorySessionService()
    USER_ID_TEAM = "user_1_agent_team"
    SESSION_ID_TEAM = "session_001_agent_team"
    await session_service_team.create_session(app_name="weather_tutorial_agent_team", user_id=USER_ID_TEAM, session_id=SESSION_ID_TEAM)
    runner_agent_team = Runner(agent=weather_agent_team, app_name="weather_tutorial_agent_team", session_service=session_service_team)
    print(f"Runner created for agent team '{runner_agent_team.agent.name}'.")

    await call_agent_async("Hello there!", runner_agent_team, USER_ID_TEAM, SESSION_ID_TEAM)
    await call_agent_async("What is the weather in New York?", runner_agent_team, USER_ID_TEAM, SESSION_ID_TEAM)
    await call_agent_async("Thanks, bye!", runner_agent_team, USER_ID_TEAM, SESSION_ID_TEAM)

    # --- Stateful Agent with output_key ---
    session_service_stateful = InMemorySessionService()
    USER_ID_STATEFUL = "user_state_demo"
    SESSION_ID_STATEFUL = "session_state_demo_001"
    initial_state = {"user_preference_temperature_unit": "Celsius"}
    await session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID_STATEFUL,
        session_id=SESSION_ID_STATEFUL,
        state=initial_state,
    )
    print(f"✅ Stateful session '{SESSION_ID_STATEFUL}' created for user '{USER_ID_STATEFUL}'.")

    greeting_agent, farewell_agent = make_sub_agents()
    root_agent_stateful = Agent(
        name="weather_agent_v4_stateful",
        model="gemini-2.5-flash",
        description="Main agent: Provides weather (state-aware unit), delegates greetings/farewells, saves report to state.",
        instruction=(
            "You are the main Weather Agent. Your job is to provide weather using 'get_weather_stateful'. "
            "The tool will format temperature based on user preference stored in state. "
            "Delegate greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
            "Handle only weather requests, greetings, and farewells."
        ),
        tools=[get_weather_stateful],
        sub_agents=[greeting_agent, farewell_agent],
        output_key="last_weather_report",
    )
    runner_root_stateful = Runner(agent=root_agent_stateful, app_name=APP_NAME, session_service=session_service_stateful)
    print(f"✅ Runner created for stateful root agent '{runner_root_stateful.agent.name}'.")

    await call_agent_async("What's the weather in London?", runner_root_stateful, USER_ID_STATEFUL, SESSION_ID_STATEFUL)

    stored_session = session_service_stateful.sessions[APP_NAME][USER_ID_STATEFUL][SESSION_ID_STATEFUL]
    stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"
    print(f"--- Stored session state updated to Fahrenheit ---")

    await call_agent_async("Tell me the weather in New York.", runner_root_stateful, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
    await call_agent_async("Hi!", runner_root_stateful, USER_ID_STATEFUL, SESSION_ID_STATEFUL)

    final_session = await session_service_stateful.get_session(app_name=APP_NAME, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    if final_session:
        print("\n--- Final Session State ---")
        print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
        print(f"Final Last Weather Report: {final_session.state.get('last_weather_report', 'Not Set')}")
        print(f"Final Last City Checked: {final_session.state.get('last_city_checked_stateful', 'Not Set')}")

    # --- Model Input Guardrail ---
    greeting_agent, farewell_agent = make_sub_agents()
    root_agent_model_guardrail = Agent(
        name="weather_agent_v5_model_guardrail",
        model="gemini-2.5-flash",
        description="Main agent: Handles weather, delegates greetings/farewells, includes input keyword guardrail.",
        instruction=(
            "You are the main Weather Agent. Provide weather using 'get_weather_stateful'. "
            "Delegate greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
            "Handle only weather requests, greetings, and farewells."
        ),
        tools=[get_weather_stateful],
        sub_agents=[greeting_agent, farewell_agent],
        output_key="last_weather_report",
        before_model_callback=block_keyword_guardrail,
    )
    runner_root_model_guardrail = Runner(agent=root_agent_model_guardrail, app_name=APP_NAME, session_service=session_service_stateful)
    print(f"✅ Runner created for model guardrail agent '{runner_root_model_guardrail.agent.name}'.")

    await call_agent_async("What is the weather in London?", runner_root_model_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
    await call_agent_async("BLOCK the request for weather in Tokyo", runner_root_model_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
    await call_agent_async("Hello again", runner_root_model_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)

    final_session = await session_service_stateful.get_session(app_name=APP_NAME, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    if final_session:
        print("\n--- Guardrail Session State ---")
        print(f"Guardrail Triggered Flag: {final_session.state.get('guardrail_block_keyword_triggered', 'Not Set (or False)')}")
        print(f"Last Weather Report: {final_session.state.get('last_weather_report', 'Not Set')}")
        print(f"Temperature Unit: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")

    # --- Tool Argument Guardrail ---
    greeting_agent, farewell_agent = make_sub_agents()
    root_agent_tool_guardrail = Agent(
        name="weather_agent_v6_tool_guardrail",
        model="gemini-2.5-flash",
        description="Main agent: Handles weather, delegates, includes input and tool guardrails.",
        instruction=(
            "You are the main Weather Agent. Provide weather using 'get_weather_stateful'. "
            "Delegate greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
            "Handle only weather, greetings, and farewells."
        ),
        tools=[get_weather_stateful],
        sub_agents=[greeting_agent, farewell_agent],
        output_key="last_weather_report",
        before_model_callback=block_keyword_guardrail,
        before_tool_callback=block_paris_tool_guardrail,
    )
    runner_root_tool_guardrail = Runner(agent=root_agent_tool_guardrail, app_name=APP_NAME, session_service=session_service_stateful)
    print(f"✅ Runner created for tool guardrail agent '{runner_root_tool_guardrail.agent.name}'.")

    await call_agent_async("What's the weather in New York?", runner_root_tool_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
    await call_agent_async("How about Paris?", runner_root_tool_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)
    await call_agent_async("Tell me the weather in London.", runner_root_tool_guardrail, USER_ID_STATEFUL, SESSION_ID_STATEFUL)

    final_session = await session_service_stateful.get_session(app_name=APP_NAME, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    if final_session:
        print("\n--- Final Tool Guardrail State ---")
        print(f"Tool Guardrail Triggered Flag: {final_session.state.get('guardrail_tool_block_triggered', 'Not Set (or False)')}")
        print(f"Last Weather Report: {final_session.state.get('last_weather_report', 'Not Set')}")
        print(f"Temperature Unit: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")


if __name__ == "__main__":
    asyncio.run(main())

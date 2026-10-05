# Build agent.py with static weather/vacation data and write it to disk.
# mlflow.pyfunc.log_model(python_model="agent.py") will package this file.

agent_code = '''
import datetime
import mlflow
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

# ─── Configured LLM endpoint ────────────────────────────────
LLM_ENDPOINT = "__LLM_ENDPOINT__"

# ─── Month lookup ────────────────────────────────────────────
MONTH_NAMES = {
    1: "January",  2: "February",  3: "March",     4: "April",
    5: "May",      6: "June",      7: "July",       8: "August",
    9: "September",10: "October", 11: "November",  12: "December",
}

# ─── Tool 1 static data: monthly weather per city ────────────
WEATHER_DATA = {
    "paris": {
        1: "Cold and rainy (avg 7C/45F)",
        2: "Cold with occasional sunshine (avg 8C/46F)",
        3: "Mild and partly cloudy (avg 11C/52F)",
        4: "Mild and pleasant (avg 15C/59F)",
        5: "Warm and sunny (avg 19C/66F)",
        6: "Warm and sunny (avg 22C/72F)",
        7: "Hot and sunny (avg 25C/77F)",
        8: "Hot and sunny (avg 24C/75F)",
        9: "Warm and pleasant (avg 21C/70F)",
        10: "Mild and rainy (avg 16C/61F)",
        11: "Cold and rainy (avg 10C/50F)",
        12: "Cold and foggy (avg 7C/45F)",
    },
    "dubai": {
        1: "Pleasant and sunny (avg 20C/68F)",
        2: "Warm and sunny (avg 21C/70F)",
        3: "Warm and sunny (avg 24C/75F)",
        4: "Hot and sunny (avg 29C/84F)",
        5: "Very hot and humid (avg 35C/95F)",
        6: "Extremely hot and humid (avg 38C/100F)",
        7: "Extremely hot and humid (avg 41C/106F)",
        8: "Extremely hot and humid (avg 40C/104F)",
        9: "Very hot and humid (avg 37C/99F)",
        10: "Hot and sunny (avg 32C/90F)",
        11: "Warm and pleasant (avg 26C/79F)",
        12: "Pleasant and sunny (avg 22C/72F)",
    },
    "mumbai": {
        1: "Pleasant and sunny (avg 24C/75F)",
        2: "Pleasant and sunny (avg 25C/77F)",
        3: "Warm and sunny (avg 28C/82F)",
        4: "Hot and humid (avg 31C/88F)",
        5: "Very hot and humid (avg 33C/91F)",
        6: "Monsoon with heavy rain (avg 29C/84F)",
        7: "Monsoon with heavy rain (avg 28C/82F)",
        8: "Monsoon with heavy rain (avg 28C/82F)",
        9: "Post-monsoon and cloudy (avg 29C/84F)",
        10: "Warm and partly cloudy (avg 31C/88F)",
        11: "Pleasant and sunny (avg 28C/82F)",
        12: "Pleasant and sunny (avg 25C/77F)",
    },
    "hyderabad": {
        1: "Pleasant and sunny (avg 22C/72F)",
        2: "Warm and sunny (avg 25C/77F)",
        3: "Hot and dry (avg 30C/86F)",
        4: "Very hot and dry (avg 35C/95F)",
        5: "Very hot and dry (avg 38C/100F)",
        6: "Hot with monsoon onset (avg 33C/91F)",
        7: "Warm and rainy (avg 27C/81F)",
        8: "Warm and rainy (avg 27C/81F)",
        9: "Post-monsoon and cloudy (avg 28C/82F)",
        10: "Warm and pleasant (avg 28C/82F)",
        11: "Pleasant and sunny (avg 24C/75F)",
        12: "Pleasant and sunny (avg 21C/70F)",
    },
}

# ─── Tool 2 static data: best vacation months per city ────────
VACATION_DATA = {
    "paris": {
        "best_months": [4, 5, 6, 9],
        "description": (
            "April to June offers pleasant weather, blooming gardens, and a lively cultural scene. "
            "September is warm with smaller crowds and a charming harvest atmosphere."
        ),
        "avoid_months": [11, 12, 1],
        "avoid_reason": "Winter is cold, grey, and rainy with limited outdoor activities.",
    },
    "dubai": {
        "best_months": [11, 12, 1, 2, 3],
        "description": (
            "November through March brings mild temperatures perfect for beaches, "
            "desert safaris, and outdoor dining."
        ),
        "avoid_months": [6, 7, 8, 9],
        "avoid_reason": "Summer (June-September) is extremely hot (40C+) and humid, making outdoor activities very uncomfortable.",
    },
    "mumbai": {
        "best_months": [11, 12, 1, 2],
        "description": (
            "November to February offers cool, dry weather ideal for exploring the city, "
            "beaches, and the vibrant street food scene."
        ),
        "avoid_months": [6, 7, 8],
        "avoid_reason": "Monsoon season (June-August) brings heavy flooding and major travel disruptions.",
    },
    "hyderabad": {
        "best_months": [10, 11, 12, 1, 2],
        "description": (
            "October to February is the most comfortable time to explore historic sites, "
            "the old city, and the legendary Hyderabadi cuisine."
        ),
        "avoid_months": [4, 5],
        "avoid_reason": "April and May are extremely hot (38-42C), making outdoor sightseeing very uncomfortable.",
    },
}


# ═══ Tool 1: get_current_weather ══════════════════════════════
@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a travel destination based on the current calendar month.
    Use this tool when the user asks about the weather or climate in a specific city.
    Supported locations: Paris, Dubai, Mumbai, Hyderabad.
    """
    current_month = datetime.datetime.now().month
    loc_key = location.strip().lower()
    if loc_key in WEATHER_DATA:
        weather    = WEATHER_DATA[loc_key][current_month]
        month_name = MONTH_NAMES[current_month]
        return f"Weather in {location.title()} this month ({month_name}): {weather}."
    supported = ", ".join(k.title() for k in WEATHER_DATA)
    return f"No weather data for '{location}'. Supported cities: {supported}."


# ═══ Tool 2: get_vacation_suggestions ════════════════════════
@tool
def get_vacation_suggestions(location: str) -> str:
    """
    Get the best and worst months for vacation planning at a given destination.
    Use this tool when the user asks about the best time to visit or plan a trip.
    Supported locations: Paris, Dubai, Mumbai, Hyderabad.
    """
    loc_key = location.strip().lower()
    if loc_key in VACATION_DATA:
        data  = VACATION_DATA[loc_key]
        best  = ", ".join(MONTH_NAMES[m] for m in data["best_months"])
        avoid = ", ".join(MONTH_NAMES[m] for m in data["avoid_months"])
        return (
            f"Best months to visit {location.title()}: {best}. "
            f"{data['description']} "
            f"Months to avoid: {avoid}. {data['avoid_reason']}"
        )
    supported = ", ".join(k.title() for k in VACATION_DATA)
    return f"No vacation data for '{location}'. Supported cities: {supported}."


# ═══ LangGraph ReAct Agent ════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a helpful vacation planning assistant. "
    "Help users understand current weather conditions in travel destinations "
    "and recommend the best months to plan vacations. "
    "Always use the provided tools to answer questions — never make up weather data or travel dates. "
    "Supported cities: Paris, Dubai, Mumbai, and Hyderabad."
)

llm   = ChatDatabricks(endpoint=LLM_ENDPOINT)
tools = [get_current_weather, get_vacation_suggestions]
tools_by_name  = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}

def call_tools(state: MessagesState) -> dict:
    last_msg = state["messages"][-1]
    results  = []
    for tc in last_msg.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
agent = graph.compile()

# ═══ PythonModel Wrapper for MLflow Serving ══════════════════════════════
class VacationPlannerAgent(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            messages = model_input.get("messages", [])
        elif hasattr(model_input, "iloc"):
            # pandas DataFrame input from serving endpoint
            messages = model_input.iloc[0]["messages"] if len(model_input) > 0 else []
        else:
            messages = []
        result = agent.invoke({"messages": messages})
        last_msg = result["messages"][-1]
        content = last_msg.content
        if not isinstance(content, str):
            content = str(content)
        return {"messages": [{"role": "assistant", "content": content}]}

# Register the PythonModel wrapper with MLflow for file-based serving
mlflow.models.set_model(VacationPlannerAgent())
'''.replace("__LLM_ENDPOINT__", LLM_ENDPOINT)

with open("agent.py", "w") as f:
    f.write(agent_code)

print("agent.py written successfully.")
print(f"  LLM endpoint : {LLM_ENDPOINT}")
print(f"  Tools        : get_current_weather, get_vacation_suggestions")
print(f"  Cities       : Paris, Dubai, Mumbai, Hyderabad")
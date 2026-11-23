import asyncio
import os
import certifi
from dotenv import load_dotenv

from typing import Any, TypedDict, Annotated
import operator
import uuid
import json
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command


from mcp_client import (
    tavily_mcp_search, 
    aviation_mcp_call, 
    extract_destination, 
    forecast_mcp_search, 
    weather_mcp_search
)

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

# =====================
# LLM
# =====================

llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=os.getenv("OPEN_AI_API_KEY"),
)

# =====================
# State
# =====================

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    #Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    #Orginial specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    #New Budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int
    revision_count: int

# =====================
# Shared helpers
# =====================

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]

def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    )
    return str(response.content)

def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model"""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object")

    return json.loads(text[start:end+1])

def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }

# =====================
# Supervisor Agent + Input Guardrails
# =====================

def supervisor_agent(state:TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belong to travel planning or travel 
information. Valid requests can include destination, fligths, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details 
are missing.

Return strict JSON only:
{{
    "allowed": true,
    "reason": ""
}}

User request: {query}
"""
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application."
            "Return strict JSON only.",
            guardrail_prompt
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used:{exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests."
            "Please ask about a destination, flight, hotel, weather, budget or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints":_empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare or booking advice
- hotel_agent: hotels, accommodation, neighborhoods or places to stay
- weather_agent: weather, climate, season, forecast or packing advice
- budget_agent: cost, affordability, price limits or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
    "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
    "trip_constraints": {{
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": []
    }},
    "reasoning": ""
}}

User request: {query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow"
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls
    }

# =====================
# Guardrail blocked response
# =====================

def guardrail_blocked_agents(state:TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrails."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)]
    }

# =====================
# Flight Agent
# =====================

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query: {query}
Airport Information: {airport_data}
Airline Information: {airline_data}

Generate:
 1. Likely departure airport
 2. Likely arrival airport
 3. Airlines serving this route
 4. Typical flight duration
 5. Estimated airfare range
 6. Peak season price warning
 7. Booking adive

 Return concise travel guidance.
"""

def flight_agent(state: TravelState):
    print("\n INSIDE FLIGHT AGENT\n")

    query = state["user_query"]
    try:
        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data= str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an export travel flight planner"
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight data unavialble: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendation generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =====================
# Hotel Agent
# =====================

def hotel_agent(state: TravelState):
    constraints = state.get("trip_constraints", {})

    query = f"""
Find affordable hotels for this trip.

Travel request: {state.get('user_query', '')}
Destination: {constraints.get('destination', '')}
Travel dates: Use dates from the user request.
Travelers: Use the requested group size.
Budget: {constraints.get('budget', '')}

Find hotels with:
- Official hotel or booking website
- Location
- Room prices
- Availability for the requested dates, if published
- Ratings and amenities

Return source URLs and distinguish published prices
from estimated prices.
"""

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }

# =====================
# Weather Agent
# =====================

def weather_agent(state: TravelState):
    llm_calls = state.get("llm_calls", 0)

    try:
        city = extract_destination(state["user_query"])
        llm_calls += 1

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
        Destination: {city}

        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            "Live weather information is temporarily unavailable. "
            "Provide general seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
        "llm_calls": llm_calls,
    }
# =====================
# Budget Agent - new specialist
# =====================

def budget_agent(state:TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query: {state['user_query']}

Trip Constraints: {state.get('trip_constraints', {}) }
FLight Results: {state.get('flight_results', "")}
Hotel Results: {state.get("hotel_results", "")}
Weather Results: {state.get("weather_results", "")}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.

"""

    response = llm.invoke([
    SystemMessage(
        content="You are a practical travel budget analyst."
    ),
    HumanMessage(content=prompt)
])

    return {
        "budget_results":response.content,
        "messages": [AIMessage(content="Budget assessment generrsated")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =====================
# Itinerary Agent
# =====================


def itinerary_agent(state: TravelState):
    prompt = f"""
You are an expert international travel planner.
Create a detailed, practical, day-by-day itinerary.

USER REQUEST:
{state.get('user_query', '')}

TRIP CONSTRAINTS:
{state.get('trip_constraints', {})}

FLIGHT INFORMATION:
{state.get('flight_results', '')}

HOTEL INFORMATION:
{state.get('hotel_results', '')}

WEATHER INFORMATION:
{state.get('weather_results', '')}

BUDGET ANALYSIS:
{state.get('budget_results', '')}

IMPORTANT RULES:
1. Never invent flight prices, hotel prices, ratings, availability,
   booking URLs, opening hours or attraction fees.
2. Clearly distinguish live prices from estimated prices.
3. Use actual URLs found in the supplied search results.
4. If a price or URL is unavailable, explicitly say so.
5. Keep the entire trip within the user's stated budget.
6. Distinguish per-person, per-couple and group costs.
7. Do not assume the number of travelers if it is unclear.
8. Avoid unrealistic travel schedules and unnecessary backtracking.

GENERATE THE FOLLOWING:

## 1. Trip Overview
- Destination
- Origin
- Travel dates
- Number of travelers
- Duration
- Travel style
- Budget and currency

## 2. Flight Details
- Departure and arrival airports
- Airline and flight number, if available
- Departure and arrival times, if available
- Duration and stops
- Price per person and total
- Booking URL
- Clearly label estimates where live fares are unavailable.

## 3. Hotel Recommendations
Recommend at least 3 suitable hotels.

For each hotel provide:
- Hotel name
- Location and neighborhood
- Room type
- Amenities
- Rating, if verified
- Price per night
- Total stay cost
- Booking URL
- Why it is suitable

## 4. Detailed Day-by-Day Itinerary

For EVERY day include:

### Morning
- Suggested time
- Activity and location
- Description
- Entrance fee

### Afternoon
- Suggested time
- Activity and location
- Description
- Transport method and estimated cost

### Evening
- Suggested time
- Activity and location
- Local food recommendations
- Estimated meal cost

Also include:
- Estimated travel time between activities
- Daily activity and transport budget
- Useful local tips

## 5. Transportation
- Airport transfers
- Local transportation
- Intercity transportation
- Estimated costs and booking links where available

## 6. Food Recommendations
- Local dishes
- Suggested restaurants or food markets
- Estimated meal costs
- Locations and links where available

## 7. Complete Budget Breakdown

Provide a table with:
- Flights
- Hotels
- Food
- Local transport
- Intercity transport
- Attractions
- Miscellaneous
- Emergency buffer

Show:
- Cost per person
- Cost per couple
- Total group cost

## 8. Booking Checklist
List what should be booked in advance and include available links.

## 9. Important Travel Information
Mention weather, visa requirements to verify, local currency,
safety considerations and practical travel tips.

Make this a genuinely useful travel plan, not a short destination overview.
Use INR for costs when the traveler is departing from India.
"""
    
    response = llm.invoke([
        SystemMessage(
            content="You are a meticulous professional travel itinerary planner."
        ),
        HumanMessage(content=prompt)
    ])

    approval_request = (
        "Please review the detailed itinerary, hotel recommendations, "
        "flight information and budget before approving."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =====================
# Human in the loop approval
# =====================

def human_approval_agent(state:TravelState):
    review = interrupt({
        "question": "Do you approve this itinerary?",
        "draft_itinerary": state.get("itinerary", ""),
        "approval_request": state.get("approval_request", ""),
        "selected_agents": state.get("selected_agents", []),
        "supervisor_resoning": state.get("supervisor_reasoning", ""),
        "expected_response": {
            "approved": True,
            "feedback": "Optional revision feedback",
        }}
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")]
    }

# =====================
# Final Response Agent
# =====================

def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get("human_feedback", "") or "Improve the draft before finalizing it."}
"""


    final_prompt = """
You are TripMate AI's professional travel planning assistant.

Prepare the final detailed travel plan using the information below.

USER REQUEST:
{state.get('user_query', '')}

TRIP CONSTRAINTS:
{state.get('trip_constraints', {})}

FLIGHTS:
{state.get('flight_results', '')}

HOTELS:
{state.get('hotel_results', '')}

WEATHER:
{state.get('weather_results', '')}

BUDGET:
{state.get('budget_results', '')}

DETAILED ITINERARY:
{state.get('itinerary', '')}

HUMAN REVIEW:
{review_instruction}

REQUIREMENTS:

- Preserve the detailed day-by-day schedule.
- Do not shorten the hotel recommendations.
- Preserve hotel and booking URLs as clickable Markdown links.
- Include all available prices and clearly label estimates.
- Show costs per person, per couple and for the whole group.
- Include transport, food, attraction and accommodation expenses.
- Clearly identify missing live data.
- Do not invent booking links, prices or availability.
- Check that the budget totals are mathematically consistent.
- Do not change the user's travel dates or group size.

Use these sections:

1. Trip Summary
2. Flight Options and Booking Links
3. Hotel Recommendations and Booking Links
4. Weather Information
5. Detailed Day-by-Day Itinerary
6. Local Food and Restaurant Recommendations
7. Transportation and Transfer Costs
8. Detailed Budget Breakdown
9. Booking Checklist
10. Important Travel Advice

Make the final response comprehensive and practical.
Do not turn this into a short summary.
"""


    response = llm.invoke([
        SystemMessage(content="You are professional AI travel booking assistant"),
        HumanMessage(content=final_prompt)
    ])

    return {
        "final_response":response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route

def route_after_human_approval(state: TravelState):
    if state["approved"]:
        return "approved"

    return "revise"


# =====================
# Build Graph
# =====================

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agents)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)
graph.add_edge("itinerary_agent", "human_approval")
graph.add_conditional_edges(
    "human_approval",
    route_after_human_approval,
    {
        "approved": "final_agent",
        "revise": "itinerary_agent",
    }
)
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =====================
# PostgreSQL Checkpointer
# =====================

DatabaseURL = get_database_url()

_conn = psycopg.connect(
    DatabaseURL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# =====================
# Function for FastAPI
# =====================

def _interrupt_payload(result:dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}

def _serialize_result(
        result: dict[str, Any],
        thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
    "itinerary", ""
)

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }

def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
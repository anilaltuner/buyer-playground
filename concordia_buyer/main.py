import datetime
import os

import sentence_transformers
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory, blank_memories, formative_memories
from concordia.clocks import game_clock
from concordia.language_model import gpt_model
from concordia import components as generic_components
from concordia.components import agent as agent_components
from concordia_buyer.memories import Memories
import random

GPT_API_KEY = os.environ.get('GPT_API_KEY', "")
GPT_MODEL_NAME = 'gpt-4o'

if not GPT_API_KEY:
    raise ValueError('GPT_API_KEY is required.')

model = gpt_model.GptLanguageModel(api_key=GPT_API_KEY, model_name=GPT_MODEL_NAME)

_embedder_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

START_TIME = datetime.datetime(year=2024, month=10, day=1, hour=20)
MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

clock = game_clock.MultiIntervalClock(start=START_TIME, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP])
agent_memory = associative_memory.AssociativeMemory(sentence_embedder=embedder, clock=clock.now)

ASSETS = {
    'LAPTOP': {"description": "Laptop: A portable computer for work and personal use.", "price": 1000},
    'SMARTPHONE': {"description": "Smartphone: A mobile device for communication and internet access.", "price": 800},
    'CAR': {"description": "Car: A vehicle for transportation.", "price": 25000},
    'HOUSE': {"description": "House: A residential property for living.", "price": 300000},
    'STOCK': {"description": "Stock: A share of ownership in a company.", "price": 100}
}

INITIAL_BUDGET = 50000
current_budget = INITIAL_BUDGET

memory_concatenation_component = Memories(memory=agent_memory, component_name='memories')

buyer_agent = basic_agent.BasicAgent(
    model,
    agent_name='Buyer',
    clock=clock,
    verbose=True,
    components=[memory_concatenation_component],
    update_interval=MAJOR_TIME_STEP
)

agent_config = formative_memories.AgentConfig(
    name='Buyer',
    gender='neutral',
    traits='analytical, decisive, budget-conscious'
)

blank_memory_factory = blank_memories.MemoryFactory(model=model, embedder=embedder, clock_now=clock.now)
formative_memory_factory = formative_memories.FormativeMemoryFactory(
    model=model,
    blank_memory_factory_call=blank_memory_factory.make_blank_memory
)

instructions = generic_components.constant.ConstantComponent(
    state=(
        f'The instructions for how to play the role of {agent_config.name} are '
        'as follows. This is a simulation of a buyer agent. The goal is to be '
        'realistic in making purchasing decisions. It is important to play the '
        f'role of a buyer like {agent_config.name} as accurately as possible, '
        'i.e., by responding in ways that you think it is likely a buyer would '
        'respond, taking into account all information about the assets, '
        f'{agent_config.name}\'s traits, and the current budget. Always use third-person limited perspective.'
    ),
    name='role playing instructions\n'
)

identity = agent_components.self_perception.SelfPerception(
    name=f'answer to what kind of buyer is {agent_config.name}',
    model=model,
    memory=agent_memory,
    agent_name=agent_config.name,
    clock_now=clock.now
)

observation = agent_components.observation.Observation(
    agent_name=agent_config.name,
    clock_now=clock.now,
    timeframe=clock.get_step_size(),
    memory=agent_memory
)

relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
    name='relevant memories',
    model=model,
    memory=agent_memory,
    agent_name=agent_config.name,
    components=[observation],
    clock_now=clock.now,
    num_memories_to_retrieve=20
)

# Add initial memories about individual assets and budget
for asset, details in ASSETS.items():
    agent_memory.add(text=f'Available asset for purchase: {details["description"]} Price: ${details["price"]}',
                     timestamp=START_TIME)
agent_memory.add(text=f'Initial budget: ${INITIAL_BUDGET}', timestamp=START_TIME)

clock.advance()

chat_history = []


def update_agent_memory(seller_utterance, buyer_reply):
    global current_budget
    agent_memory.add(
        text=f"Conversation: Seller: {seller_utterance}\nBuyer: {buyer_reply}",
        timestamp=datetime.datetime.now()
    )
    agent_memory.add(
        text=f"Current budget: ${current_budget}",
        timestamp=datetime.datetime.now()
    )


def process_purchase(asset):
    global current_budget
    if asset.upper() in ASSETS and current_budget >= ASSETS[asset.upper()]["price"]:
        current_budget -= ASSETS[asset.upper()]["price"]
        return f"Purchase of {asset} successful. Remaining budget: ${current_budget}"
    elif asset.upper() not in ASSETS:
        return f"Asset {asset} not available for purchase."
    else:
        return f"Insufficient funds to purchase {asset}. Current budget: ${current_budget}"


while True:
    utterance_from_seller = input("Seller: ")
    chat_history.append(f"Seller: {utterance_from_seller}")

    if utterance_from_seller.lower() in ['exit', 'quit', 'bye']:
        print("Conversation ended.")
        break

    buyer_replies = buyer_agent.say(utterance_from_seller)
    chat_history.append(f"Buyer: {buyer_replies}")

    print("Buyer:", buyer_replies)

    if "buy" in buyer_replies.lower() or "purchase" in buyer_replies.lower():
        for asset in ASSETS:
            if asset.lower() in buyer_replies.lower():
                purchase_result = process_purchase(asset)
                print(purchase_result)
                agent_memory.add(text=purchase_result, timestamp=datetime.datetime.now())

    print("\nChat History:")
    for message in chat_history:
        print(message)
    print()

    update_agent_memory(utterance_from_seller, buyer_replies)
    print(agent_memory.get_data_frame())

    # Randomly update asset prices
    for asset in ASSETS:
        price_change = random.uniform(-0.05, 0.05)  # -5% to +5% change
        ASSETS[asset]["price"] = int(ASSETS[asset]["price"] * (1 + price_change))
        agent_memory.add(
            text=f"Price update: {asset} now costs ${ASSETS[asset]['price']}",
            timestamp=datetime.datetime.now()
        )

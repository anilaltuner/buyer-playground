import datetime
import os

import sentence_transformers
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.language_model import gpt_model
from concordia import components as generic_components
from concordia.components import agent as agent_components

from concordia_buyer.memories import Memories

GPT_API_KEY = os.environ.get('GPT_API_KEY', "")
GPT_MODEL_NAME = 'gpt-4o'

if not GPT_API_KEY:
    raise ValueError('GPT_API_KEY is required.')

model = gpt_model.GptLanguageModel(api_key=GPT_API_KEY,
                                   model_name=GPT_MODEL_NAME)

_embedder_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

START_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)

MAJOR_TIME_STEP = datetime.timedelta(minutes=30)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)

clock = game_clock.MultiIntervalClock(
    start=START_TIME,
    step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP])

agent_memory = associative_memory.AssociativeMemory(
    sentence_embedder=embedder,
    clock=clock.now,
)

# Define individual assets as strings
ASSET_LAPTOP = "Laptop: A portable computer for work and personal use."
ASSET_SMARTPHONE = "Smartphone: A mobile device for communication and internet access."
ASSET_CAR = "Car: A vehicle for transportation."
ASSET_HOUSE = "House: A residential property for living."
ASSET_STOCK = "Stock: A share of ownership in a company."

memory_concatenation_component = Memories(
    memory=agent_memory,
    component_name='memories'
)

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

# First we create a new clock.
clock = game_clock.MultiIntervalClock(
    start=START_TIME,
    step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP],
)
clock.set(START_TIME)

blank_memory_factory = blank_memories.MemoryFactory(
    model=model,
    embedder=embedder,
    clock_now=clock.now,
)
formative_memory_factory = formative_memories.FormativeMemoryFactory(
    model=model,
    blank_memory_factory_call=blank_memory_factory.make_blank_memory,
)

instructions = generic_components.constant.ConstantComponent(
    state=(
        f'The instructions for how to play the role of {agent_config.name} are '
        'as follows. This is a simulation of a buyer agent. The goal is to be '
        'realistic in making purchasing decisions. It is important to play the '
        f'role of a buyer like {agent_config.name} as accurately as possible, '
        'i.e., by responding in ways that you think it is likely a buyer would '
        'respond, taking into account all information about the assets and '
        f'{agent_config.name}\'s traits. Always use third-person limited perspective.'
    ),
    name='role playing instructions\n',
)

identity = agent_components.self_perception.SelfPerception(
    name=f'answer to what kind of buyer is {agent_config.name}',
    model=model,
    memory=agent_memory,
    agent_name=agent_config.name,
    clock_now=clock.now,
)

observation = agent_components.observation.Observation(
    agent_name=agent_config.name,
    clock_now=clock.now,
    timeframe=clock.get_step_size(),
    memory=agent_memory,
)

relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
    name='relevant memories',
    model=model,
    memory=agent_memory,
    agent_name=agent_config.name,
    components=[observation],
    clock_now=clock.now,
    num_memories_to_retrieve=20,
)

agent_memory.get_data_frame()

# Add initial memories about individual assets
agent_memory.add(
    text=f'Available asset for purchase: {ASSET_LAPTOP}',
    timestamp=START_TIME,
)
agent_memory.add(
    text=f'Available asset for purchase: {ASSET_SMARTPHONE}',
    timestamp=START_TIME,
)
agent_memory.add(
    text=f'Available asset for purchase: {ASSET_CAR}',
    timestamp=START_TIME,
)
agent_memory.add(
    text=f'Available asset for purchase: {ASSET_HOUSE}',
    timestamp=START_TIME,
)
agent_memory.add(
    text=f'Available asset for purchase: {ASSET_STOCK}',
    timestamp=START_TIME,
)

agent_memory.get_data_frame()
clock.advance()

# Initialize an empty list to store chat history
chat_history = []

# Start a conversation loop
while True:
    # Get input from the seller
    utterance_from_seller = input("Seller: ")

    # Add seller's utterance to chat history
    chat_history.append(f"Seller: {utterance_from_seller}")

    # Check if the seller wants to end the conversation
    if utterance_from_seller.lower() in ['exit', 'quit', 'bye']:
        print("Conversation ended.")
        break

    # Process the input and get buyer's reply, passing the chat history
    buyer_replies = buyer_agent.say(utterance_from_seller)
    print(agent_memory.get_data_frame())

    # Add buyer's reply to chat history
    chat_history.append(f"Buyer: {buyer_replies}")

    # Print the buyer's reply
    print("Buyer:", buyer_replies)

    # Print the entire chat history
    print("\nChat History:")
    for message in chat_history:
        print(message)
    print()  # Add a blank line for readability

    # Update the buyer agent's memory with the latest conversation
    agent_memory.add(
        text=f"Conversation: {chat_history[-2]}\n{chat_history[-1]}",
        timestamp=datetime.datetime.now(),
    )

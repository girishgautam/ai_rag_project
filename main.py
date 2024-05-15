from dotenv import load_dotenv
import pandas as pd
import os
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.llms.openai import OpenAI


load_dotenv()


population_path = os.path.join('data', 'population.csv')

population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df,
                                           verbose=True,
                                           instruction_str = instruction_str)

population_query_engine.update_prompts({'pandas_prompts' : new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine = population_query_engine,
        metadata=ToolMetadata(
        name='populatio_data',
        description='this gives information about the worl population and emographics'
    ),
        )
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

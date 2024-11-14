from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, RunnerArgument, CoStormRunner
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import BingSearch
import os

# Co-STORM adopts the same multi LM system paradigm as STORM 
lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
# openai_kwargs = {
#     "api_key": os.getenv("OPENAI_API_KEY"),
#     "api_provider": "openai",
#     "temperature": 1.0,
#     "top_p": 0.9,
#     "api_base": None,
# } 
openai_kwargs = {
    "api_key": "sk-c9ruGN8RcO1ml3SsTX1m4zhvDxaANbp79KLeF3QaEUxt8YBI@23889",
    "api_provider": "openai",
    "temperature": 1.0,
    "top_p": 0.9,
    "api_base": None,
} 
gpt_4o_model_name = 'gpt-4o'
question_answering_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=1000, **openai_kwargs)
discourse_manage_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=500, **openai_kwargs)
utterance_polishing_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=2000, **openai_kwargs)
warmstart_outline_gen_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=500, **openai_kwargs)
question_asking_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=300, **openai_kwargs)
knowledge_base_lm = OpenAIModel(model=gpt_4o_model_name, max_tokens=1000, **openai_kwargs)

lm_config.set_question_answering_lm(question_answering_lm)
lm_config.set_discourse_manage_lm(discourse_manage_lm)
lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
lm_config.set_question_asking_lm(question_asking_lm)
lm_config.set_knowledge_base_lm(knowledge_base_lm)

# Check out the Co-STORM's RunnerArguments class for more configurations.
topic = input('Topic: ')
runner_argument = RunnerArgument(topic=topic)
logging_wrapper = LoggingWrapper(lm_config)
# bing_rm = BingSearch(bing_search_api_key=os.environ.get("BING_SEARCH_API_KEY"),
#                      k=runner_argument.retrieve_top_k)
bing_rm = BingSearch(bing_search_api_key=os.environ.get("BING_SEARCH_API_KEY"),
                     k=runner_argument.retrieve_top_k)
costorm_runner = CoStormRunner(lm_config=lm_config,
                               runner_argument=runner_argument,
                               logging_wrapper=logging_wrapper,
                               rm=bing_rm)
# Warm start the system to build shared conceptual space between Co-STORM and users
costorm_runner.warm_start()

# Step through the collaborative discourse 
# Run either of the code snippets below in any order, as many times as you'd like
# To observe the conversation:
conv_turn = costorm_runner.step()
# To inject your utterance to actively steer the conversation:
costorm_runner.step(user_utterance="YOUR UTTERANCE HERE")

# Generate report based on the collaborative discourse
costorm_runner.knowledge_base.reorganize()
article = costorm_runner.generate_report()
print(article)
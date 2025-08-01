# Certification Project
By Calvin Schröder

## General Decisions
As a Model for most evaluation `microsoft:phi2` was used. In initial trails the model behaved sufficiently bad to potentially see an improvement by adding better prompting, RAG and finetuning.
Additionally the small size of the model promised fast execution times on a local machine.

## Task 1
For task 1 to have a comparison model `mistral:7b` was used. The model is considerably larger than the `microsoft:phi2` model so better results are to be expected.
All system prompts specifically include the requirement for the conversation to be "face-to-face" like, since this was a concept the LLMs often ignored initially, creating output more resembling written communication like an email.
The prompts were initially hand written and later improved by assistance of gemini:2.5-Flash.

The created test complaints were created to include typical questions that also could later benefit from RAG. The evaulation included a mixture of static checks and also checks using `llm-rubic`.
For evaluation usualy the `mistral:7b` model was used. A specific case was the `llm-rubic` requiring the conversation to be "face-to-face" like here an `llama3:70b` was used since the mistral model often failed to dectect such conversations or outright flagged them the wrong way.

In the resulting testset the friendly prompt worked best on both the mistral and microsoft model. Passing 100% or 83,3% of the test respectively. Generally the `mistral:7b` responses surpased the `microsoft:phi2` responses in direct comparision with the same prompt. An expected outcome.

Suprisingly the `microsoft:phi2` consistently failed to create proper responses for some prompts. A reocuring issue persistent between runs.
The run used for this report can be found in the coresponding `results.json` file in the task1 folder.

## Task 2
For task 2 three wikipedia articles were downloaded using the respective python library. The text was cleaned and saved to txt files. An additional faq was created in a JSON format some questions specifically designed to aid specific test_cases previously designed but some encoded with different wording.

The RAG generation code is present in the folder `shared` since it is reused for later task 3 and 5.sentence_transformers was used to create the vector store. A different setup from the exercise was used to later simply retrieve the retrieval scores.

For evaluation again promptfoo was used, but without automatic validation instead they were handchecked.
This included also checking the RAG retrieval.
First a k=3 was used retrieving 3 documents. This often lead to Wikipedia chunks beeing retrieved which had a bad impact on the models performance. The small `microsoft:phi2` models seemed to be overloaded and started hallucinating a lot having the actual task outside of the context window. Setting k=2 lead to no wikipedia articles beeing retrieved but less hallucinating in any testcase which was unfortunate.

Even with less input the models still sometimes hallucinated. Two specific hallucinations were common:
- Generating and solving Riddles. This for example happend on the question "My internet does not work at all." on the "empathic" prompt.
- Generating full conversations. This for example happend on the structured prompt even without context on the "My internet does not work at all." question.

The retrival generally retrieved helpful faq questions specifically the ones linked to specific questions. This also improved the responses at times like in the "My internet speed is much slower than advertised." case on the empathic prompt. But did not translate to the answers at all times.

In general the impact of RAG was there visible, further improvements could be made to reduce k to k=1 to see if the issues with hallucinations would be resolved then.

## Task 3
For task 3 a threshold of 0.6 for retrival was used after some experimentation.
In the end RAG was used in 4 out of 6 test cases. 
The generated `agent_decision_log.json` includes the responses for all the prompts. Of course the decision to use RAG or not remains the same independently of the system prompt.

## Task 4
For task 4 the HH and Alpaca Dataset was brought into a common format for which only the "chosen" field of the HH dataset was used.
I used a random_search between different parameter combinations to generate 5 different models. This setup would theoretically work for all or at least more models but since finetuning a single model on this setup took around 4 hours on my machine (even after reducing the dataset size) 5 models seemed like a good compromise. The setup evaluates the model using a 80/20 train test split of the data and automatically choses the best one.

## Task 5
In Task 5 first all responses for the combinations where generated.
BertScore was used to compare the impact the customization did to the output (highest f1).
There RAG was closest to the prompt-only responses while the finetuned and agent models where further away from the reference answer which the prompt-only responses represented.
Again this is to be expected considering the agent model 
The detailed results are available in `bertscore_results.json`.

For the llm evaluation again `llama3:70b` was used. It evaluated the responses of each setup based on helpfulness, clarity, empathy and safety on a 1-5 scale. Here the RAG version archived the highest results and finetuned only the lowest. Looking at the more detailed results the helpfulness and clarity hold the finetuned model back. However the finetuned model shows the highest safety of all setups.

The agent version improves the finetuned model in clarity without being more helpful and archives similar scores as the prompt only setup due to higher safety and clarity.
From that it can be assumed that the finetuning improved safety significantly.
Notably this experiment is limited since a single pass through the very limited dataset was used.

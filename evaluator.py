import os
from langchain.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load documents
loader = DirectoryLoader("./data")
documents = loader.load()

# Generate synthetic test data
generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(documents, test_size=30, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# Export to Pandas DataFrame for easy handling
test_df = testset.to_pandas()

# Extract questions and answers
test_questions = test_df['question'].values.tolist()
test_answers = [[item] for item in test_df['answer'].values.tolist()]

metrics = [faithfulness, answer_relevancy, answer_correctness]
# Assuming `results` is the output of your RAG pipeline evaluation
results = evaluate(your_dataset, metrics=metrics)

# Analyze results
# Compare the performance of different models based on the evaluation metrics

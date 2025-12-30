Support Ticket Auto-Tagging System

Objective This project automates the classification of customer support tickets using a Large Language Model (LLM). The focus is on transforming messy, free-text data into structured categories to streamline support workflows.

Multi-Class Ranking: Extract the top 3 most probable tags for every ticket.

Accuracy Optimization: Compare and improve performance using Prompt Engineering and Few-Shot Learning.

Comparative Analysis: Evaluate the difference between Zero-Shot and Few-Shot results side-by-side.

Methodology The system uses a modular Python architecture to process data and interact with the LLM.

Data Ingestion: A custom loader merges ticket subjects and descriptions into a single context.

Prompt Engineering: System prompts are designed to enforce a strict JSON output format for reliable parsing.

Learning Strategies:

Zero-Shot: Direct classification based on the model's general training.

Few-Shot: Providing 3-5 specific examples within the prompt to align the model with specific company labels.

Inference Engine: Utilizing Llama-3.3-70b via the Groq API for high-speed processing.

Key Results and Observations Zero-Shot: Provided accurate general themes but lacked consistent label naming (e.g., mixing "WiFi" and "Network").

Few-Shot: Significantly improved label consistency and justification quality by following the provided examples.

Ranking: Successfully identified multi-dimensional issues by providing primary, secondary, and tertiary tags.

How to Run the System

Prerequisites Ensure you have Python installed and a Groq API Key.

Installation Clone the repository and install the required dependencies:

pip install streamlit pandas groq python-dotenv

Environment Setup Create a .env file in the root directory and add your API key:
GROQ_API_KEY=your_key_here

Launch the Application
Run the Streamlit dashboard:

streamlit run app.py


<body>

<h1>📘 RAG Assistant (Retrieval Augmented Generation)</h1>
<p>
A powerful interactive RAG application built with 
<strong>Streamlit</strong>, <strong>LangChain</strong>, <strong>LangGraph</strong>,
<strong>Groq LLaMA 3.3</strong>, and <strong>ChromaDB</strong>.
Users can upload <strong>PDF</strong> or <strong>TXT</strong> files and ask questions,
and the agent responds using both <strong>document retrieval</strong> and optional 
<strong>web search</strong> tools.
</p>

<hr>

<h2>🚀 Features</h2>
<ul>
    <li>📄 Upload <strong>PDF</strong> or <strong>TXT</strong> files</li>
    <li>🔍 Content is split into chunks and indexed using <strong>Chroma VectorStore</strong></li>
    <li>🧠 <strong>Custom retriever tool</strong> for answering from uploaded documents</li>
    <li>🌐 Optional <strong>web search tool</strong> for external information</li>
    <li>🤖 <strong>Groq LLaMA 3.3</strong> for ultra-fast inference</li>
    <li>🔁 <strong>LangGraph</strong> tool-calling workflow (LLM → Tool → LLM)</li>
    <li>💬 Clean chat UI using Streamlit</li>
    <li>🗑️ One-click <strong>Clear Chat History</strong></li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
├── app.py                   # Streamlit UI
├── rag_model.py             # RAG model with LangGraph + tools
├── requirements.txt         # Dependencies
└── README.html              # This file
</pre>

<hr>

<h2>⚙️ How It Works</h2>

<h3>1️⃣ File Upload</h3>
<p>User uploads a PDF or TXT file. The file is temporarily saved and passed into:</p>

<pre><code>build_rag_agent_from_file(file_path)</code></pre>

<h3>2️⃣ Text Processing</h3>
<ul>
    <li>Loads the document</li>
    <li>Splits into chunks (RecursiveCharacterTextSplitter)</li>
    <li>Creates embeddings (MiniLM-L6-v2)</li>
    <li>Stores chunks in <strong>Chroma</strong> (in-memory)</li>
</ul>

<h3>3️⃣ RAG Tools</h3>
<ul>
    <li><strong>retriever_tool</strong> → Retrieves PDF/TXT chunks</li>
    <li><strong>web_search_tool</strong> → (Optional) Tavily Search API</li>
</ul>

<h3>4️⃣ LangGraph Execution Flow</h3>

<pre>
User Message
    ↓
LLM Node (call_llm)
    ↓
Should Continue?
    ↓ YES (LLM requested a tool)
Tool Node (take_action)
    ↓
LLM
    ↓
Final Answer
</pre>

<hr>

<h2>💻 Running the App</h2>

<h3>1️⃣ Install dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>2️⃣ Run Streamlit</h3>
<pre><code>streamlit run app.py</code></pre>

<h3>3️⃣ Upload file and chat</h3>
- Upload a PDF or TXT document  
- Ask questions  
- RAG Agent retrieves knowledge  
- Groq LLM generates answers  

<hr>

<h2>📦 Example Code Snippet (Streamlit)</h2>

<pre><code>
if uploaded_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    rag_agent = build_rag_agent_from_file(temp_path)

result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
</code></pre>

<hr>

<h2>📊 RAG Graph Diagram</h2>

<pre>
                [User Question]
                        |
                        v
               ┌─────────────────┐
               │  LLM Node       │
               │  (call_llm)     │
               └──────┬──────────┘
                      |
            Should Continue?
                YES |  NO
                      v
               ┌─────────────────┐
               │  Tool Node      │
               │ (take_action)   │
               └──────┬──────────┘
                      |
         ┌────────────┴────────────┐
         |                           |
 [retriever_tool]           [web_search_tool]
         |                           |
         └────────────┬──────────────┘
                      |
                      v
               ┌─────────────────┐
               │  LLM Node       │
               └──────┬──────────┘
                      |
                   [Answer]
</pre>

<hr>

<h2>🔧 Requirements</h2>
<p>Create a <code>requirements.txt</code> file with:</p>

<pre><code>
streamlit
langchain
langchain-community
langchain-core
langgraph
langchain-groq
chromadb
sentence-transformers
pypdf
python-dotenv
tavily-python
</code></pre>

<hr>

<h2>📜 License</h2>
<p>This project is open-source and free to modify.</p>

</body>
</html>

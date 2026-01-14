# 🏢 CorporateMind AI: Multi-Agent Enterprise Assistant
An advanced Multi-Agent system designed for corporate environments, capable of intelligently routing and resolving employee queries across HR and IT domains using **LangGraph** and **Llama-3**.

---

## 🚀 Project Overview
This project moves beyond simple RAG chatbots to a **Multi-Agent Architecture**. It simulates a professional workplace environment where a **Supervisor Agent** analyzes user intent and routes the query to the correct specialized department (HR Specialist or IT Support).

### Key Features:
- **Multi-Agent Orchestration:** Powered by `LangGraph` for stateful, multi-turn conversations.
- **Intelligent Routing:** A Supervisor Agent acting as a router to prevent prompt injection and ensure high accuracy.
- **Optimized LLM:** Built on **Llama-3-8B-Instruct** using **4-bit Quantization (bitsandbytes)** for efficient GPU performance.
- **Production-Ready State Management:** Uses a unified `AgentState` to track messages and routing decisions across the graph.

---

## 🏗️ System Architecture
The system follows a **Graph-based workflow**:
1. **START**: User inputs a query.
2. **Supervisor (The Router)**: Classifies intent (HR, IT, or General).
3. **Condition Edges**: 
   - If **HR**: Routes to `HR_Specialist_Node`.
   - If **IT**: Routes to `IT_Specialist_Node`.
4. **END**: Final response is cleaned and delivered.

---

## 🧠 Core Logic & Routing Mechanism
This system operates on a **Stateful Graph Architecture**, ensuring high-precision routing:

*   **Intent Classification:** Uses **Llama-3** as a reasoning engine to classify queries into specific routing tokens (`TO_HR`, `TO_IT`).
*   **Stateful Communication:** Uses `AgentState` to maintain conversation memory across different specialized agents.
*   **Specialized Nodes:** Dedicated environments for HR policies and IT troubleshooting, ensuring domain-specific accuracy.
*   **Output Refinement:** A custom parsing layer removes technical tokens, delivering a polished, enterprise-ready response.

---

## 🛠️ Tech Stack
- **Frameworks:** LangChain, LangGraph.
- **LLM:** Llama-3-8B-Instruct (4-bit Quantized).
- **Optimization:** BitsAndBytes, Accelerate.
- **Environment:** Kaggle/Colab (GPU T4).

---

## 💻 Sample Output
> **User:** "My laptop screen is flickering, what should I do?"  
> **Process:** `Supervisor` -> `IT_Specialist`  
> **AI:** "IT Specialist: Please visit the IT portal to open a hardware ticket or visit the IT desk on Floor 2."

---

## 👨‍💻 Author
**Abdul Rahman Ahmed**
*Junior Data Scientist & AI Engineer*

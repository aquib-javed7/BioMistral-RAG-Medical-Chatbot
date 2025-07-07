# 🩺 BioMistral Medical RAG Chatbot

A **GenAI-powered Retrieval-Augmented Generation (RAG) chatbot** that uses the open-source **BioMistral-7B GGUF model** to answer medical questions from uploaded PDF documents like clinical reports, guidelines, and research materials.

> 💡 Built with 🧠 LLM (BioMistral), 🗃️ LangChain, 📚 FAISS, 🧬 HuggingFace Embeddings, and 🚀 Streamlit.

---

## 📌 Features

- Upload any medical PDF file (e.g. Oncology reports, HIV/AIDS UN booklet)
- Ask medical questions from the document
- Uses **LangChain**'s RAG pipeline for context-aware QA
- Integrates **BioMistral-7B** model locally via `llama-cpp-python`
- Fully interactive via **Streamlit UI**
- Loads model via HuggingFace Hub on first run

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── README.md # Project description (this file)
└── /model/ # (Optional) Local model directory if hosting offline

---

## 🧠 Model Used

- 🔬 `BioMistral-7B.Q4_K_M.gguf` — from Hugging Face [QuantFactory/BioMistral-7B-GGUF](https://huggingface.co/QuantFactory/BioMistral-7B-GGUF)
- Model type: **Quantized GGUF (Q4_K_M)** — optimized for local CPU-based inference using `llama-cpp-python`

---

## 📄 PDF Examples Used

- **Oncology.pdf** – Breast Cancer Pathology, Treatment, MDT, Physiotherapy, etc.
- **jc306-un-staff-rev1_en.pdf** – HIV/AIDS Guidelines for UN Staff and Families

---


## 📦 Dependencies
See requirements.txt. Key packages:
streamlit
langchain
langchain-community
llama-cpp-python
faiss-cpu
pypdf
sentence-transformers
huggingface_hub

---


🧪 LLM + RAG Flow

PDF ➜ Split Pages ➜ FAISS Vector Store ➜ HuggingFace Embeddings
        ⇩                       ⇩
  Query Box → LangChain RAG → BioMistral LLM → Answer

---

📜 License

This project is intended for educational and research use only. Please consult healthcare professionals before taking any action based on generated answers.

---

🙋‍♂️ Author

Akib Javith V S T

Data Scientist | GenAI Explorer

[LinkedIn](https://www.linkedin.com/in/akib-javith-37bbbb324/) | [GitHub](https://github.com/aquib-javed7) | [HuggingFace](https://huggingface.co/aquibjaved7)



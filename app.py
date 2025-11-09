
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS MPS quirks

# import torch
# import streamlit as st
# from io import BytesIO, StringIO
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from pdfminer.high_level import extract_text as pdfminer_extract_text

# # NEW: for timing + logs
# import time, json
# from datetime import datetime

# # LangChain (modern)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from langchain.chains.summarize import load_summarize_chain

# # Local transformers pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# # Your HTML templates
# from htmlTemplates import css, bot_template, user_template


# # -----------------------------
# # Robust PDF text extraction
# # -----------------------------
# def get_pdf_text(pdf_docs):
#     """
#     Robust text extraction:
#       1) Try PyPDF2 for embedded text
#       2) Fallback to pdfminer.six (handles more layouts)
#       3) If still empty, flag as OCR candidate (commented OCR block included)
#     """
#     total_text = []
#     ocr_candidates = 0

#     for up in (pdf_docs or []):
#         data = up.read()
#         if not data:
#             continue

#         # --- Try PyPDF2 ---
#         chunk = ""
#         try:
#             reader = PdfReader(BytesIO(data))
#             for page in reader.pages:
#                 chunk += page.extract_text() or ""
#         except Exception:
#             pass

#         # --- Fallback: pdfminer.six ---
#         if len(chunk.strip()) < 30:
#             try:
#                 chunk = pdfminer_extract_text(BytesIO(data)) or ""
#             except Exception:
#                 pass

#         # --- Mark for OCR if still empty ---
#         if len(chunk.strip()) < 30:
#             ocr_candidates += 1

#             # OPTIONAL OCR (requires Tesseract & pypdfium2)
#             # import pypdfium2, pytesseract
#             # images = []
#             # pdf = pypdfium2.PdfDocument(BytesIO(data))
#             # for i in range(len(pdf)):
#             #     page = pdf.get_page(i)
#             #     bmp = page.render(scale=2).to_pil()
#             #     images.append(bmp)
#             # page_text = [pytesseract.image_to_string(img) for img in images]
#             # chunk = "\n".join(page_text)

#         total_text.append(chunk)

#     text = "\n".join(total_text)
#     if not text.strip():
#         st.error(
#             "No extractable text found. This PDF is likely scanned or image-based. "
#             "Enable OCR in the code (comments) or run the file through an OCR tool like `ocrmypdf`."
#         )
#     elif ocr_candidates:
#         st.warning(
#             f"{ocr_candidates} file(s) had little/no embedded text. If answers look empty, enable OCR in the code or pre-OCR your PDFs."
#         )

#     return text


# # -----------------------------
# # Chunking (fit T5's context)
# # -----------------------------
# def get_text_chunks(text: str):
#     splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", " ", ""],
#         chunk_size=500,     # small to fit T5's ~512-token window w/ prompt
#         chunk_overlap=96,
#         length_function=len,
#     )
#     return splitter.split_text(text)


# # -----------------------------
# # Embeddings (cached)
# # -----------------------------
# @st.cache_resource
# def load_embedder():
#     """Cache the embedder so Streamlit hot-reloads don't re-download the model."""
#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# # -----------------------------
# # Prompts
# # -----------------------------
# QA_PROMPT = PromptTemplate.from_template(
#     "Answer the user's question **only** using the context. "
#     "If the answer isn't in the context, say: \"I couldn't find that in the documents.\" "
#     "Be concise (2–4 sentences).\n\n"
#     "Context:\n{context}\n\n"
#     "Question: {question}\n"
#     "Answer:"
# )

# CONDENSE_PROMPT = PromptTemplate.from_template(
#     "Rewrite the follow-up question to be standalone using the chat history.\n"
#     "Chat history:\n{chat_history}\n\n"
#     "Follow-up question: {question}\n"
#     "Standalone question:"
# )


# # -----------------------------
# # Local FLAN-T5 (cached)
# # -----------------------------
# @st.cache_resource
# def load_local_llm():
#     """
#     Try FLAN-T5-LARGE first (≈770M params).
#     If memory issues happen, fall back to FLAN-T5-BASE automatically.
#     """
#     preferred = "google/flan-t5-large"
#     fallback = "google/flan-t5-base"

#     # Device + dtype (fp16 on MPS/CUDA; fp32 on CPU)
#     if torch.backends.mps.is_available():
#         device = "mps"
#         dtype = torch.float16
#     elif torch.cuda.is_available():
#         device = "cuda"
#         dtype = torch.float16
#     else:
#         device = "cpu"
#         dtype = torch.float32

#     def _make_pipeline(model_id: str):
#         tok = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_id,
#             dtype=(dtype if device != "cpu" else torch.float32),  # use dtype (not torch_dtype)
#             low_cpu_mem_usage=True,
#         )
#         if device != "cpu":
#             model.to(device)
#         gen = pipeline(
#             task="text2text-generation",
#             model=model,
#             tokenizer=tok,
#             max_new_tokens=192,      # leave room for input
#             min_new_tokens=48,
#             do_sample=False,         # deterministic, better for QA
#             num_beams=6,             # quality bump
#             length_penalty=1.05,
#             no_repeat_ngram_size=3,
#         )
#         return HuggingFacePipeline(pipeline=gen)

#     try:
#         llm = _make_pipeline(preferred)
#         st.caption(f"LLM ready: local {preferred} (text2text-generation)")
#         return llm
#     except Exception as e:
#         st.warning(f"Couldn't load {preferred} ({e}). Falling back to {fallback}...")
#         llm = _make_pipeline(fallback)
#         st.caption(f"LLM ready: local {fallback} (text2text-generation)")
#         return llm


# # -----------------------------
# # Vector store + retriever
# # -----------------------------
# def get_vectorstore(text_chunks):
#     embeddings = load_embedder()
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# # -----------------------------
# # Conversational QA chain
# # -----------------------------
# def get_conversation_chain(vectorstore):
#     llm = load_local_llm()

#     # Use MMR to keep 3–4 diverse, short chunks in context
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
#     )

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer",
#     )

#     # IMPORTANT: return sources so we can detect empty retrieval and fallback
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT},
#         condense_question_prompt=CONDENSE_PROMPT,
#         return_source_documents=True,
#         output_key="answer",   # keep memory happy
#     )
#     return chain


# # -----------------------------
# # One-click summarization
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def _summarizer_llm():
#     # reuse the same local model for summarization chain
#     return load_local_llm()

# def summarize_all_chunks(text_chunks):
#     """Map-reduce summary over the whole document using the same local LLM."""
#     llm = _summarizer_llm()
#     docs = [Document(page_content=t) for t in text_chunks]
#     chain = load_summarize_chain(llm, chain_type="map_reduce")
#     return chain.run(docs)


# # -----------------------------
# # NON-DESTRUCTIVE extractor test button
# # -----------------------------
# def test_extract_and_chunk(pdf_docs):
#     """Run extraction on copies of uploads (does not consume them)."""
#     total_text = []
#     ocr_candidates = 0

#     for up in (pdf_docs or []):
#         data = up.getvalue()  # <--- IMPORTANT: use getvalue(), do NOT consume .read()
#         if not data:
#             continue

#         chunk = ""
#         # Try PyPDF2
#         try:
#             reader = PdfReader(BytesIO(data))
#             for page in reader.pages:
#                 chunk += page.extract_text() or ""
#         except Exception:
#             pass

#         # Fallback pdfminer
#         if len(chunk.strip()) < 30:
#             try:
#                 chunk = pdfminer_extract_text(BytesIO(data)) or ""
#             except Exception:
#                 pass

#         if len(chunk.strip()) < 30:
#             ocr_candidates += 1

#         total_text.append(chunk)

#     text = "\n".join(total_text)
#     chunks = get_text_chunks(text)

#     # Simple metrics
#     total_chars = len(text)
#     num_chunks = len(chunks)
#     avg_len = (sum(len(c) for c in chunks) / max(1, num_chunks)) if num_chunks else 0
#     longest = sorted((len(c) for c in chunks), reverse=True)[:5]

#     # UI: show metrics + samples
#     st.subheader("Extraction & Chunking — Test Results")
#     st.write(f"- **Total characters extracted:** {total_chars}")
#     st.write(f"- **Number of chunks:** {num_chunks}")
#     st.write(f"- **Average chunk length (chars):** {int(avg_len)}")
#     st.write(f"- **Top 5 longest chunks (chars):** {longest}")

#     if num_chunks:
#         with st.expander("Preview: first 2 chunks"):
#             for i, c in enumerate(chunks[:2], start=1):
#                 st.markdown(f"**Chunk {i}**  \nLength: {len(c)}")
#                 st.code(c[:1000])  # cap display

#     # Downloads: extracted text + chunks (one per line separated by delimiter)
#     st.download_button(
#         "Download extracted_text.txt",
#         data=text,
#         file_name="extracted_text.txt",
#         mime="text/plain",
#     )
#     st.download_button(
#         "Download chunks_debug.txt",
#         data="\n\n----- CHUNK DELIMITER -----\n\n".join(chunks),
#         file_name="chunks_debug.txt",
#         mime="text/plain",
#     )

#     if ocr_candidates:
#         st.warning(
#             f"{ocr_candidates} file(s) had little/no embedded text. "
#             "If results look empty, enable OCR in code or pre-OCR your PDFs."
#         )


# # -----------------------------
# # NEW: Retrieval add-ons (preview, latency logs, mini demo)
# # -----------------------------
# def _mk_demo_retriever():
#     """Create a retriever matching your chain settings, without touching chain logic."""
#     vs = st.session_state.get("vectorstore")
#     if not vs:
#         st.warning("Vector index is not ready yet. Click Process first.")
#         return None
#     return vs.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
#     )

# def _time_call(fn, *args, **kwargs):
#     t0 = time.perf_counter()
#     out = fn(*args, **kwargs)
#     dt = (time.perf_counter() - t0) * 1000.0  # ms
#     return out, dt

# def _score_grounding(answer, retrieved_docs):
#     # 0–2 scale: 2 if we have citations and a non-trivial answer, 1 if only one is decent, else 0
#     has_cite = bool(retrieved_docs)
#     good_len = len((answer or "").strip()) >= 40
#     if has_cite and good_len:
#         return 2
#     if has_cite or good_len:
#         return 1
#     return 0

# def run_mini_demo(queries, k=3):
#     """
#     For each query:
#       - show top-k retrieved chunks,
#       - call your existing chain (so logic stays the same),
#       - show the answer + citations,
#       - log timing to CSV/JSON.
#     """
#     retriever = _mk_demo_retriever()
#     chain = st.session_state.get("conversation")
#     if not retriever or not chain:
#         st.warning("Please Process PDFs first.")
#         return

#     log_rows = []
#     now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

#     for idx, q in enumerate(queries, 1):
#         q = (q or "").strip()
#         if not q:
#             continue

#         # 1) Retrieval timing
#         docs, t_retrieval = _time_call(retriever.get_relevant_documents, q)
#         top_docs = docs[:k]

#         # 2) LLM timing (use your existing chain so logic stays identical)
#         resp, t_llm = _time_call(chain.invoke, {"question": q})
#         answer = resp.get("answer", "")
#         cites = resp.get("source_documents") or []

#         # 3) Render preview
#         st.markdown(f"### Demo Query {idx}")
#         st.write(f"**Query:** {q}")

#         with st.expander("Top retrieved snippets"):
#             for i, d in enumerate(top_docs, 1):
#                 text = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"**{i}.** {text[:500]}{'…' if len(text)>500 else ''}")

#         st.markdown("**Answer:**")
#         st.write(answer if answer else "_(empty)_")

#         if cites:
#             st.markdown("**Citations:**")
#             for i, d in enumerate(cites, 1):
#                 text = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"[{i}] {text[:400]}{'…' if len(text)>400 else ''}")
#         else:
#             st.caption("No citations returned.")

#         # 4) Timing / rubric
#         t_total = t_retrieval + t_llm
#         score = _score_grounding(answer, cites)
#         st.caption(f"⏱ retrieval: {t_retrieval:.1f} ms | LLM: {t_llm:.1f} ms | total: {t_total:.1f} ms | grounding score: {score}/2")
#         st.markdown("---")

#         # 5) Log row
#         log_rows.append({
#             "timestamp": now,
#             "query": q,
#             "retrieval_ms": round(t_retrieval, 1),
#             "rerank_ms": 0.0,            # no reranker in this pipeline
#             "llm_ms": round(t_llm, 1),
#             "total_ms": round(t_total, 1),
#             "citations": len(cites),
#             "grounding_score_0_2": score,
#         })

#     if log_rows:
#         # CSV
#         csv_buf = StringIO()
#         keys = ["timestamp","query","retrieval_ms","rerank_ms","llm_ms","total_ms","citations","grounding_score_0_2"]
#         csv_buf.write(",".join(keys) + "\n")
#         for r in log_rows:
#             csv_buf.write(",".join(str(r[k]) for k in keys) + "\n")
#         st.download_button("Download latency log (CSV)", data=csv_buf.getvalue(),
#                            file_name="latency_log.csv", mime="text/csv")

#         # JSON
#         st.download_button("Download latency log (JSON)",
#                            data=json.dumps(log_rows, indent=2),
#                            file_name="latency_log.json",
#                            mime="application/json")


# # -----------------------------
# # Summary intent detector
# # -----------------------------
# def _looks_like_summary_request(q: str) -> bool:
#     ql = (q or "").lower()
#     return any(
#         key in ql
#         for key in ["what is this about", "what is this pdf about", "summarize", "summary", "overview"]
#     )


# # -----------------------------
# # UI: normal chat handler (unchanged logic; added citations view)
# # -----------------------------
# def handle_userinput(user_question):
#     # call with invoke() to avoid deprecation warnings
#     resp = st.session_state.conversation.invoke({"question": user_question})
#     st.session_state.chat_history = resp.get("chat_history", [])
#     answer = resp.get("answer", "")
#     sources = resp.get("source_documents") or []

#     # If user asked for a summary/what-is-this-about AND retrieval was empty,
#     # automatically fall back to a whole-PDF summary.
#     retrieval_empty = (len(sources) == 0) or ("I couldn't find that in the documents." in answer)
#     if _looks_like_summary_request(user_question) and retrieval_empty:
#         if st.session_state.get("text_chunks"):
#             with st.spinner("Summarizing the whole PDF..."):
#                 answer = summarize_all_chunks(st.session_state.text_chunks)

#     # render chat history
#     for i, message in enumerate(st.session_state.chat_history):
#         html = user_template if i % 2 == 0 else bot_template
#         st.write(html.replace("{{MSG}}", message.content), unsafe_allow_html=True)

#     # show citations (retrieved snippets)
#     if sources:
#         with st.expander("Citations (retrieved snippets)"):
#             for i, d in enumerate(sources, 1):
#                 preview = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"**[{i}]** {preview[:320]}{'…' if len(preview)>320 else ''}")

#     # also render the (possibly replaced) answer explicitly if history didn’t include it
#     if answer and (not st.session_state.chat_history or st.session_state.chat_history[-1].content != answer):
#         st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


# # -----------------------------
# # Streamlit app
# # -----------------------------
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
#     st.write(css, unsafe_allow_html=True)

#     # Reset to avoid stale chains after code/config changes
#     with st.sidebar:
#         if st.button("Reset app"):
#             st.session_state.clear()
#             st.experimental_rerun()

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#     if "text_chunks" not in st.session_state:
#         st.session_state.text_chunks = None

#     st.header("Chat with multiple PDFs 📚")

#     # Main question box
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)

#     # Summarize button (works after Process)
#     if st.session_state.get("text_chunks") and st.button("Summarize this PDF"):
#         with st.spinner("Summarizing..."):
#             summary = summarize_all_chunks(st.session_state.text_chunks)
#         st.subheader("Summary")
#         st.write(summary)

#     # Sidebar upload & processing
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'",
#             accept_multiple_files=True,
#             type=["pdf"],
#         )

#         # Quick, non-destructive test button
#         if st.button("Test extraction & chunking", disabled=not pdf_docs):
#             with st.spinner("Running tests..."):
#                 test_extract_and_chunk(pdf_docs)

#         if st.button("Process", disabled=not pdf_docs):
#             with st.spinner("Processing..."):
#                 # 1) Extract text robustly
#                 raw_text = get_pdf_text(pdf_docs)

#                 # 2) Split into chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # 3) Show indexing stats (sanity check)
#                 total_chars = sum(len(c) for c in text_chunks)
#                 st.info(f"Indexed {len(text_chunks)} chunks (~{total_chars} characters).")
#                 if total_chars < 50:
#                     st.error(
#                         "Very little/no text was indexed. This PDF may be scanned. "
#                         "Enable OCR in the code or pre-OCR your PDFs and try again."
#                     )
#                     st.stop()

#                 # Stash chunks for summarization
#                 st.session_state.text_chunks = text_chunks

#                 # 4) Build vector store
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.vectorstore = vectorstore  # <--- ADDED: so demos can use it

#                 # 5) Build conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#                 st.success("Ready! Ask a question above or click 'Summarize this PDF'.")

#         # NEW: Mini retrieval demo panel
#         st.subheader("Mini Retrieval Demo")
#         demo_qs_text = st.text_area(
#             "Enter 3–5 queries (one per line)",
#             value="What is this PDF about?\nList the core concepts.\nGive two key definitions.\nWhere is the method explained?\nSummarize section 1."
#         )
#         demo_k = st.slider("Show top-k retrieved snippets", 1, 5, 3)
#         if st.button("Run retrieval demo", disabled=not st.session_state.get("conversation")):
#             with st.spinner("Running mini demo..."):
#                 run_mini_demo([q for q in demo_qs_text.splitlines() if q.strip()], k=demo_k)

#     if not st.session_state.conversation:
#         st.info("Upload PDFs and click **Process** to start.")


# if __name__ == "__main__":
#     main()
# ==== keep TensorFlow/Keras out of Transformers (harmless even though we no longer use HF locally) ====
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS MPS quirks

# import torch
# import streamlit as st
# from io import BytesIO, StringIO
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from pdfminer.high_level import extract_text as pdfminer_extract_text

# # NEW: for timing + logs
# import time, json
# from datetime import datetime

# # LangChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from langchain.chains.summarize import load_summarize_chain

# # >>> OpenAI (cheapest solid chat model)
# from langchain_openai import ChatOpenAI

# # Your HTML templates
# from htmlTemplates import css, bot_template, user_template


# # -----------------------------
# # Robust PDF text extraction
# # -----------------------------
# def get_pdf_text(pdf_docs):
#     """
#     Robust text extraction:
#       1) Try PyPDF2 for embedded text
#       2) Fallback to pdfminer.six (handles more layouts)
#       3) If still empty, flag as OCR candidate (commented OCR block included)
#     """
#     total_text = []
#     ocr_candidates = 0

#     for up in (pdf_docs or []):
#         data = up.read()
#         if not data:
#             continue

#         # --- Try PyPDF2 ---
#         chunk = ""
#         try:
#             reader = PdfReader(BytesIO(data))
#             for page in reader.pages:
#                 chunk += page.extract_text() or ""
#         except Exception:
#             pass

#         # --- Fallback: pdfminer.six ---
#         if len(chunk.strip()) < 30:
#             try:
#                 chunk = pdfminer_extract_text(BytesIO(data)) or ""
#             except Exception:
#                 pass

#         # --- Mark for OCR if still empty ---
#         if len(chunk.strip()) < 30:
#             ocr_candidates += 1

#             # OPTIONAL OCR (requires Tesseract & pypdfium2)
#             # import pypdfium2, pytesseract
#             # images = []
#             # pdf = pypdfium2.PdfDocument(BytesIO(data))
#             # for i in range(len(pdf)):
#             #     page = pdf.get_page(i)
#             #     bmp = page.render(scale=2).to_pil()
#             #     images.append(bmp)
#             # page_text = [pytesseract.image_to_string(img) for img in images]
#             # chunk = "\n".join(page_text)

#         total_text.append(chunk)

#     text = "\n".join(total_text)
#     if not text.strip():
#         st.error(
#             "No extractable text found. This PDF is likely scanned or image-based. "
#             "Enable OCR in the code (comments) or run the file through an OCR tool like `ocrmypdf`."
#         )
#     elif ocr_candidates:
#         st.warning(
#             f"{ocr_candidates} file(s) had little/no embedded text. If answers look empty, enable OCR in the code or pre-OCR your PDFs."
#         )

#     return text


# # -----------------------------
# # Chunking (small to fit context)
# # -----------------------------
# def get_text_chunks(text: str):
#     splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", " ", ""],
#         chunk_size=500,
#         chunk_overlap=96,
#         length_function=len,
#     )
#     return splitter.split_text(text)


# # -----------------------------
# # Embeddings (cached)
# # -----------------------------
# @st.cache_resource
# def load_embedder():
#     """Cache the embedder so Streamlit hot-reloads don't re-download the model."""
#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# # -----------------------------
# # Prompts (softened for chat tasks)
# # -----------------------------
# QA_PROMPT = PromptTemplate.from_template(
#     "Use the context to answer. If something is not in the context, say: "
#     "\"I couldn't find that in the documents.\" Be concise (2–4 sentences). "
#     "If the user asks for help generating tasks (like quiz questions or an outline) "
#     "based on the document, you may do so using the context.\n\n"
#     "Context:\n{context}\n\n"
#     "Question: {question}\n"
#     "Answer:"
# )

# CONDENSE_PROMPT = PromptTemplate.from_template(
#     "Rewrite the follow-up question to be standalone using the chat history.\n"
#     "Chat history:\n{chat_history}\n\n"
#     "Follow-up question: {question}\n"
#     "Standalone question:"
# )


# # -----------------------------
# # OpenAI LLM (cached)
# # -----------------------------
# @st.cache_resource
# def load_llm_openai():
#     # reads OPENAI_API_KEY from environment (.env loaded in main())
#     return ChatOpenAI(
#         model="gpt-4o-mini",  # low-cost + capable
#         temperature=0,        # deterministic for factual QA
#         max_tokens=256,
#     )


# # -----------------------------
# # Vector store + retriever
# # -----------------------------
# def get_vectorstore(text_chunks):
#     embeddings = load_embedder()
#     return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# # -----------------------------
# # Conversational QA chain
# # -----------------------------
# def get_conversation_chain(vectorstore):
#     llm = load_llm_openai()

#     # Use MMR to keep diverse, short chunks in context
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
#     )

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer",
#     )

#     # Return source documents so we can show citations + do fallbacks
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT},
#         condense_question_prompt=CONDENSE_PROMPT,
#         return_source_documents=True,
#         output_key="answer",
#     )
#     return chain


# # -----------------------------
# # One-click summarization
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def _summarizer_llm():
#     # reuse the same OpenAI model for summarization chain
#     return load_llm_openai()

# def summarize_all_chunks(text_chunks):
#     """Map-reduce summary over the whole document using the same LLM."""
#     llm = _summarizer_llm()
#     docs = [Document(page_content=t) for t in text_chunks]
#     chain = load_summarize_chain(llm, chain_type="map_reduce")
#     return chain.run(docs)


# # -----------------------------
# # NON-DESTRUCTIVE extractor test button
# # -----------------------------
# def test_extract_and_chunk(pdf_docs):
#     """Run extraction on copies of uploads (does not consume them)."""
#     total_text = []
#     ocr_candidates = 0

#     for up in (pdf_docs or []):
#         data = up.getvalue()  # do NOT consume .read()
#         if not data:
#             continue

#         chunk = ""
#         try:
#             reader = PdfReader(BytesIO(data))
#             for page in reader.pages:
#                 chunk += page.extract_text() or ""
#         except Exception:
#             pass

#         if len(chunk.strip()) < 30:
#             try:
#                 chunk = pdfminer_extract_text(BytesIO(data)) or ""
#             except Exception:
#                 pass

#         if len(chunk.strip()) < 30:
#             ocr_candidates += 1

#         total_text.append(chunk)

#     text = "\n".join(total_text)
#     chunks = get_text_chunks(text)

#     # Simple metrics
#     total_chars = len(text)
#     num_chunks = len(chunks)
#     avg_len = (sum(len(c) for c in chunks) / max(1, num_chunks)) if num_chunks else 0
#     longest = sorted((len(c) for c in chunks), reverse=True)[:5]

#     # UI: show metrics + samples
#     st.subheader("Extraction & Chunking — Test Results")
#     st.write(f"- **Total characters extracted:** {total_chars}")
#     st.write(f"- **Number of chunks:** {num_chunks}")
#     st.write(f"- **Average chunk length (chars):** {int(avg_len)}")
#     st.write(f"- **Top 5 longest chunks (chars):** {longest}")

#     if num_chunks:
#         with st.expander("Preview: first 2 chunks"):
#             for i, c in enumerate(chunks[:2], start=1):
#                 st.markdown(f"**Chunk {i}**  \nLength: {len(c)}")
#                 st.code(c[:1000])

#     st.download_button(
#         "Download extracted_text.txt",
#         data=text,
#         file_name="extracted_text.txt",
#         mime="text/plain",
#     )
#     st.download_button(
#         "Download chunks_debug.txt",
#         data="\n\n----- CHUNK DELIMITER -----\n\n".join(chunks),
#         file_name="chunks_debug.txt",
#         mime="text/plain",
#     )

#     if ocr_candidates:
#         st.warning(
#             f"{ocr_candidates} file(s) had little/no embedded text. "
#             "If results look empty, enable OCR in code or pre-OCR your PDFs."
#         )


# # -----------------------------
# # Retrieval demo helpers
# # -----------------------------
# def _mk_demo_retriever():
#     vs = st.session_state.get("vectorstore")
#     if not vs:
#         st.warning("Vector index is not ready yet. Click Process first.")
#         return None
#     return vs.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
#     )

# def _time_call(fn, *args, **kwargs):
#     t0 = time.perf_counter()
#     out = fn(*args, **kwargs)
#     dt = (time.perf_counter() - t0) * 1000.0  # ms
#     return out, dt

# def _score_grounding(answer, retrieved_docs):
#     has_cite = bool(retrieved_docs)
#     good_len = len((answer or "").strip()) >= 40
#     if has_cite and good_len:
#         return 2
#     if has_cite or good_len:
#         return 1
#     return 0

# def run_mini_demo(queries, k=3):
#     retriever = _mk_demo_retriever()
#     chain = st.session_state.get("conversation")
#     if not retriever or not chain:
#         st.warning("Please Process PDFs first.")
#         return

#     log_rows = []
#     now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

#     for idx, q in enumerate(queries, 1):
#         q = (q or "").strip()
#         if not q:
#             continue

#         # 1) Retrieval timing
#         docs, t_retrieval = _time_call(retriever.get_relevant_documents, q)
#         top_docs = docs[:k]

#         # 2) LLM timing (same chain)
#         resp, t_llm = _time_call(chain.invoke, {"question": q})
#         answer = resp.get("answer", "")
#         cites = resp.get("source_documents") or []

#         # 3) Render
#         st.markdown(f"### Demo Query {idx}")
#         st.write(f"**Query:** {q}")

#         with st.expander("Top retrieved snippets"):
#             for i, d in enumerate(top_docs, 1):
#                 text = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"**{i}.** {text[:500]}{'…' if len(text)>500 else ''}")

#         st.markdown("**Answer:**")
#         st.write(answer if answer else "_(empty)_")

#         if cites:
#             st.markdown("**Citations:**")
#             for i, d in enumerate(cites, 1):
#                 text = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"[{i}] {text[:400]}{'…' if len(text)>400 else ''}")
#         else:
#             st.caption("No citations returned.")

#         # 4) Timing / rubric
#         t_total = t_retrieval + t_llm
#         score = _score_grounding(answer, cites)
#         st.caption(f"⏱ retrieval: {t_retrieval:.1f} ms | LLM: {t_llm:.1f} ms | total: {t_total:.1f} ms | grounding score: {score}/2")
#         st.markdown("---")

#         # 5) Log row
#         log_rows.append({
#             "timestamp": now,
#             "query": q,
#             "retrieval_ms": round(t_retrieval, 1),
#             "rerank_ms": 0.0,
#             "llm_ms": round(t_llm, 1),
#             "total_ms": round(t_total, 1),
#             "citations": len(cites),
#             "grounding_score_0_2": score,
#         })

#     if log_rows:
#         # CSV
#         csv_buf = StringIO()
#         keys = ["timestamp","query","retrieval_ms","rerank_ms","llm_ms","total_ms","citations","grounding_score_0_2"]
#         csv_buf.write(",".join(keys) + "\n")
#         for r in log_rows:
#             csv_buf.write(",".join(str(r[k]) for k in keys) + "\n")
#         st.download_button("Download latency log (CSV)", data=csv_buf.getvalue(),
#                            file_name="latency_log.csv", mime="text/csv")

#         # JSON
#         st.download_button("Download latency log (JSON)",
#                            data=json.dumps(log_rows, indent=2),
#                            file_name="latency_log.json",
#                            mime="application/json")


# # -----------------------------
# # Intent detectors + quiz generator
# # -----------------------------
# def _looks_like_summary_request(q: str) -> bool:
#     ql = (q or "").lower().strip()
#     summary_keys = [
#         "what is this pdf about",
#         "what's this pdf about",
#         "what is this about",
#         "what's this about",
#         "summarize", "summary", "overview",
#         "give me an overview", "tl;dr"
#     ]
#     return any(k in ql for k in summary_keys)

# def _looks_like_quiz_request(q: str) -> bool:
#     ql = (q or "").lower().strip()
#     quiz_keys = [
#         "ask me questions", "quiz me", "practice questions",
#         "make questions", "create questions", "flashcards",
#         "can you ask me", "test me"
#     ]
#     return any(k in ql for k in quiz_keys)

# def _generate_quiz_from_chunks(chunks, llm, n=6):
#     sample = sorted(chunks, key=len, reverse=True)[:6]
#     context = "\n\n---\n\n".join(sample)
#     prompt = (
#         "You are a helpful tutor. Using ONLY the context, create "
#         f"{n} short, clear questions that test understanding. "
#         "Prefer concrete details/definitions over trivia. "
#         "Number them 1..n. Do NOT include answers.\n\n"
#         f"Context:\n{context}\n\nQuestions:"
#     )
#     return llm.invoke(prompt).content


# # -----------------------------
# # UI: normal chat handler (summary & quiz aware)
# # -----------------------------
# def handle_userinput(user_question):
#     # 0) Detect special intents up front
#     if _looks_like_quiz_request(user_question):
#         if st.session_state.get("text_chunks"):
#             quiz = _generate_quiz_from_chunks(
#                 st.session_state.text_chunks,
#                 load_llm_openai(),
#                 n=6
#             )
#             st.write(bot_template.replace("{{MSG}}", quiz), unsafe_allow_html=True)
#             return
#         else:
#             st.warning("Please upload & Process PDFs first so I can make questions.")
#             return

#     # 1) Run the normal RAG chain
#     resp = st.session_state.conversation.invoke({"question": user_question})
#     st.session_state.chat_history = resp.get("chat_history", [])
#     answer = (resp.get("answer") or "")
#     sources = resp.get("source_documents") or []

#     # 2) Robust retrieval quality check
#     total_chars = sum(len((d.page_content or "").strip()) for d in sources)
#     couldnt = "couldn't find" in answer.lower()
#     retrieval_empty = (total_chars < 40) or couldnt

#     # 3) Summary fallback if intent + weak retrieval
#     if _looks_like_summary_request(user_question) and retrieval_empty:
#         if st.session_state.get("text_chunks"):
#             with st.spinner("Summarizing the whole PDF..."):
#                 answer = summarize_all_chunks(st.session_state.text_chunks)
#             st.caption("🔎 Retrieval looked sparse, so I generated a whole-PDF summary.")

#     # 4) Render prior messages
#     for i, message in enumerate(st.session_state.chat_history):
#         html = user_template if i % 2 == 0 else bot_template
#         st.write(html.replace("{{MSG}}", message.content), unsafe_allow_html=True)

#     # 5) Citations (only if we actually have useful snippets)
#     if sources and total_chars >= 40:
#         with st.expander("Citations (retrieved snippets)"):
#             for i, d in enumerate(sources, 1):
#                 preview = (d.page_content or "").strip().replace("\n", " ")
#                 st.markdown(f"**[{i}]** {preview[:320]}{'…' if len(preview)>320 else ''}")
#     else:
#         st.caption("No useful snippets retrieved.")

#     # 6) Explicitly show the (possibly replaced) answer
#     if answer and (not st.session_state.chat_history or st.session_state.chat_history[-1].content != answer):
#         st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


# # -----------------------------
# # Streamlit app
# # -----------------------------
# def main():
#     load_dotenv()  # <-- loads OPENAI_API_KEY
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
#     st.write(css, unsafe_allow_html=True)

#     # Reset to avoid stale chains after code/config changes
#     with st.sidebar:
#         if st.button("Reset app"):
#             st.session_state.clear()
#             st.experimental_rerun()

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#     if "text_chunks" not in st.session_state:
#         st.session_state.text_chunks = None

#     st.header("Chat with multiple PDFs 📚")

#     # Main question box
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)

#     # Summarize button (works after Process)
#     if st.session_state.get("text_chunks") and st.button("Summarize this PDF"):
#         with st.spinner("Summarizing..."):
#             summary = summarize_all_chunks(st.session_state.text_chunks)
#         st.subheader("Summary")
#         st.write(summary)

#     # Sidebar upload & processing
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'",
#             accept_multiple_files=True,
#             type=["pdf"],
#         )

#         # Quick, non-destructive test button
#         if st.button("Test extraction & chunking", disabled=not pdf_docs):
#             with st.spinner("Running tests..."):
#                 test_extract_and_chunk(pdf_docs)

#         if st.button("Process", disabled=not pdf_docs):
#             with st.spinner("Processing..."):
#                 # 1) Extract text robustly
#                 raw_text = get_pdf_text(pdf_docs)

#                 # 2) Split into chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # 3) Show indexing stats (sanity check)
#                 total_chars = sum(len(c) for c in text_chunks)
#                 st.info(f"Indexed {len(text_chunks)} chunks (~{total_chars} characters).")
#                 if total_chars < 50:
#                     st.error(
#                         "Very little/no text was indexed. This PDF may be scanned. "
#                         "Enable OCR in the code or pre-OCR your PDFs and try again."
#                     )
#                     st.stop()

#                 # Stash chunks for summarization & quiz
#                 st.session_state.text_chunks = text_chunks

#                 # 4) Build vector store
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.vectorstore = vectorstore  # for mini demo

#                 # 5) Build conversation chain (OpenAI)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#                 st.success("Ready! Ask a question above, say 'summarize', or 'ask me questions'.")

#         # Mini retrieval demo panel
#         st.subheader("Mini Retrieval Demo")
#         demo_qs_text = st.text_area(
#             "Enter 3–5 queries (one per line)",
#             value="What is this PDF about?\nList the core concepts.\nGive two key definitions.\nWhere is the method explained?\nSummarize section 1."
#         )
#         demo_k = st.slider("Show top-k retrieved snippets", 1, 5, 3)
#         if st.button("Run retrieval demo", disabled=not st.session_state.get("conversation")):
#             with st.spinner("Running mini demo..."):
#                 run_mini_demo([q for q in demo_qs_text.splitlines() if q.strip()], k=demo_k)

#     if not st.session_state.conversation:
#         st.info("Upload PDFs and click **Process** to start.")


# if __name__ == "__main__":
#     main()

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # quiet HF fork warning

import torch
import streamlit as st
from io import BytesIO, StringIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

import time, json
from datetime import datetime

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# OpenAI (cheap, capable chat model)
from langchain_openai import ChatOpenAI

# Your HTML templates
from htmlTemplates import css, bot_template, user_template


# -----------------------------
# Robust PDF text extraction
# -----------------------------
def get_pdf_text(pdf_docs):
    total_text = []
    ocr_candidates = 0

    for up in (pdf_docs or []):
        data = up.read()
        if not data:
            continue

        chunk = ""
        try:
            reader = PdfReader(BytesIO(data))
            for page in reader.pages:
                chunk += page.extract_text() or ""
        except Exception:
            pass

        if len(chunk.strip()) < 30:
            try:
                chunk = pdfminer_extract_text(BytesIO(data)) or ""
            except Exception:
                pass

        if len(chunk.strip()) < 30:
            ocr_candidates += 1

        total_text.append(chunk)

    text = "\n".join(total_text)
    if not text.strip():
        st.error(
            "No extractable text found. This PDF is likely scanned or image-based. "
            "Enable OCR or run it through a tool like `ocrmypdf`."
        )
    elif ocr_candidates:
        st.warning(
            f"{ocr_candidates} file(s) had little/no embedded text. Consider OCR if answers look empty."
        )
    return text


# -----------------------------
# Chunking
# -----------------------------
def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=96,
        length_function=len,
    )
    return splitter.split_text(text)


# -----------------------------
# Embeddings (cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------
# Prompts
# -----------------------------
QA_PROMPT = PromptTemplate.from_template(
    "Use the context to answer. If something is not in the context, say: "
    "\"I couldn't find that in the documents.\" Be concise (2–4 sentences). "
    "If the user asks for help generating tasks (quiz questions/outline/steps), you may do so using the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

CONDENSE_PROMPT = PromptTemplate.from_template(
    "Rewrite the follow-up question to be standalone using the chat history.\n"
    "Chat history:\n{chat_history}\n\n"
    "Follow-up question: {question}\n"
    "Standalone question:"
)


# -----------------------------
# OpenAI LLM (cached)
# -----------------------------
@st.cache_resource
def load_llm_openai():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=256,
    )


# -----------------------------
# Vector store + retriever
# -----------------------------
def get_vectorstore(text_chunks):
    embeddings = load_embedder()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# -----------------------------
# Conversational QA chain
# -----------------------------
def get_conversation_chain(vectorstore):
    llm = load_llm_openai()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_PROMPT,
        return_source_documents=True,
        output_key="answer",
    )
    return chain


# -----------------------------
# Summarization
# -----------------------------
@st.cache_resource(show_spinner=False)
def _summarizer_llm():
    return load_llm_openai()

def summarize_all_chunks(text_chunks):
    llm = _summarizer_llm()
    docs = [Document(page_content=t) for t in text_chunks]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    # Use invoke to avoid deprecation; handle return type across LC versions
    out = chain.invoke({"input_documents": docs})
    return out.get("output_text") if isinstance(out, dict) else str(out)


# -----------------------------
# Non-destructive extractor test
# -----------------------------
def test_extract_and_chunk(pdf_docs):
    total_text = []
    ocr_candidates = 0

    for up in (pdf_docs or []):
        data = up.getvalue()
        if not data:
            continue

        chunk = ""
        try:
            reader = PdfReader(BytesIO(data))
            for page in reader.pages:
                chunk += page.extract_text() or ""
        except Exception:
            pass

        if len(chunk.strip()) < 30:
            try:
                chunk = pdfminer_extract_text(BytesIO(data)) or ""
            except Exception:
                pass

        if len(chunk.strip()) < 30:
            ocr_candidates += 1

        total_text.append(chunk)

    text = "\n".join(total_text)
    chunks = get_text_chunks(text)

    total_chars = len(text)
    num_chunks = len(chunks)
    avg_len = (sum(len(c) for c in chunks) / max(1, num_chunks)) if num_chunks else 0
    longest = sorted((len(c) for c in chunks), reverse=True)[:5]

    st.subheader("Extraction & Chunking — Test Results")
    st.write(f"- **Total characters extracted:** {total_chars}")
    st.write(f"- **Number of chunks:** {num_chunks}")
    st.write(f"- **Average chunk length (chars):** {int(avg_len)}")
    st.write(f"- **Top 5 longest chunks (chars):** {longest}")

    if num_chunks:
        with st.expander("Preview: first 2 chunks"):
            for i, c in enumerate(chunks[:2], start=1):
                st.markdown(f"**Chunk {i}**  \nLength: {len(c)}")
                st.code(c[:1000])

    st.download_button("Download extracted_text.txt", data=text,
                       file_name="extracted_text.txt", mime="text/plain")
    st.download_button("Download chunks_debug.txt",
                       data="\n\n----- CHUNK DELIMITER -----\n\n".join(chunks),
                       file_name="chunks_debug.txt", mime="text/plain")

    if ocr_candidates:
        st.warning(
            f"{ocr_candidates} file(s) had little/no embedded text. Consider OCR if results look empty."
        )


# -----------------------------
# Retrieval demo helpers
# -----------------------------
def _mk_demo_retriever():
    vs = st.session_state.get("vectorstore")
    if not vs:
        st.warning("Vector index is not ready yet. Click Process first.")
        return None
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 24, "lambda_mult": 0.3},
    )

def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt

def _score_grounding(answer, retrieved_docs):
    has_cite = bool(retrieved_docs)
    good_len = len((answer or "").strip()) >= 40
    if has_cite and good_len:
        return 2
    if has_cite or good_len:
        return 1
    return 0

def run_mini_demo(queries, k=3):
    retriever = _mk_demo_retriever()
    chain = st.session_state.get("conversation")
    if not retriever or not chain:
        st.warning("Please Process PDFs first.")
        return

    log_rows = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for idx, q in enumerate(queries, 1):
        q = (q or "").strip()
        if not q:
            continue

        docs, t_retrieval = _time_call(retriever.get_relevant_documents, q)
        top_docs = docs[:k]
        resp, t_llm = _time_call(chain.invoke, {"question": q})
        answer = resp.get("answer", "")
        cites = resp.get("source_documents") or []

        st.markdown(f"### Demo Query {idx}")
        st.write(f"**Query:** {q}")

        with st.expander("Top retrieved snippets"):
            for i, d in enumerate(top_docs, 1):
                text = (d.page_content or "").strip().replace("\n", " ")
                st.markdown(f"**{i}.** {text[:500]}{'…' if len(text)>500 else ''}")

        st.markdown("**Answer:**")
        st.write(answer if answer else "_(empty)_")

        if cites:
            st.markdown("**Citations:**")
            for i, d in enumerate(cites, 1):
                text = (d.page_content or "").strip().replace("\n", " ")
                st.markdown(f"[{i}] {text[:400]}{'…' if len(text)>400 else ''}")
        else:
            st.caption("No citations returned.")

        t_total = t_retrieval + t_llm
        score = _score_grounding(answer, cites)
        st.caption(f"⏱ retrieval: {t_retrieval:.1f} ms | LLM: {t_llm:.1f} ms | total: {t_total:.1f} ms | grounding score: {score}/2")
        st.markdown("---")

        log_rows.append({
            "timestamp": now,
            "query": q,
            "retrieval_ms": round(t_retrieval, 1),
            "rerank_ms": 0.0,
            "llm_ms": round(t_llm, 1),
            "total_ms": round(t_total, 1),
            "citations": len(cites),
            "grounding_score_0_2": score,
        })

    if log_rows:
        csv_buf = StringIO()
        keys = ["timestamp","query","retrieval_ms","rerank_ms","llm_ms","total_ms","citations","grounding_score_0_2"]
        csv_buf.write(",".join(keys) + "\n")
        for r in log_rows:
            csv_buf.write(",".join(str(r[k]) for k in keys) + "\n")
        st.download_button("Download latency log (CSV)", data=csv_buf.getvalue(),
                           file_name="latency_log.csv", mime="text/csv")
        st.download_button("Download latency log (JSON)",
                           data=json.dumps(log_rows, indent=2),
                           file_name="latency_log.json",
                           mime="application/json")


# -----------------------------
# Intent detectors + quiz + steps + small-talk
# -----------------------------
def _looks_like_summary_request(q: str) -> bool:
    ql = (q or "").lower().strip()
    summary_keys = [
        "summarize", "summary", "summery", "tl;dr", "overview",
        "summerize", "sumarize", "summerise", "summarise"
    ]
    generic_intro = [
        "what is this pdf", "what's this pdf", "what is this about",
        "what's this about", "explain", "explanation", "describe", "give me an overview"
    ]
    return any(k in ql for k in summary_keys + generic_intro)

def _looks_like_quiz_request(q: str) -> bool:
    ql = (q or "").lower().strip()
    keys = [
        "ask me questions", "quiz me", "practice questions",
        "make questions", "create questions", "flashcards",
        "can you ask me", "test me"
    ]
    return any(k in ql for k in keys)

def _looks_like_steps_request(q: str) -> bool:
    ql = (q or "").lower().strip()
    keys = [
        "steps", "how do i do this", "how to do this", "what are the steps",
        "guide me", "walk me through", "instructions", "process", "plan"
    ]
    return any(k in ql for k in keys)

def _is_smalltalk(q: str) -> bool:
    ql = (q or "").lower().strip()
    return any(k in ql for k in ["thanks", "thank you", "thx", "ok", "okay", "cool", "got it"])


def _generate_quiz_from_chunks(chunks, llm, n=6):
    sample = sorted(chunks, key=len, reverse=True)[:6]
    context = "\n\n---\n\n".join(sample)
    prompt = (
        "You are a helpful tutor. Using ONLY the context, create "
        f"{n} short, clear questions that test understanding. "
        "Prefer concrete details/definitions over trivia. "
        "Number them 1..n. Do NOT include answers.\n\n"
        f"Context:\n{context}\n\nQuestions:"
    )
    return llm.invoke(prompt).content

def _generate_steps_from_chunks(chunks, llm, n=6):
    sample = sorted(chunks, key=len, reverse=True)[:6]
    context = "\n\n---\n\n".join(sample)
    prompt = (
        "Using ONLY the context, write a numbered step-by-step plan to complete the assignment. "
        "Keep it concise (5–9 steps), action-oriented, and specific to what's in the context. "
        "If something is not in the context, do not invent it.\n\n"
        f"Context:\n{context}\n\nSteps:"
    )
    return llm.invoke(prompt).content


# -----------------------------
# UI chat helpers
# -----------------------------
def _ensure_chat_store():
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []  # list[(role, content)]
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []
    if "last_used_summary_fallback" not in st.session_state:
        st.session_state.last_used_summary_fallback = False

def _render_ui_messages():
    for role, msg in st.session_state.ui_messages:
        html = user_template if role == "human" else bot_template
        st.write(html.replace("{{MSG}}", msg), unsafe_allow_html=True)


# -----------------------------
# Chat handler (clean, UI-managed history)
# -----------------------------
def handle_userinput(user_question):
    _ensure_chat_store()

    q = (user_question or "").strip()
    if not q:
        return

    # Append user message to UI history
    st.session_state.ui_messages.append(("human", q))

    # Small talk: reply without hitting RAG
    if _is_smalltalk(q):
        ack = "You're welcome! Want a summary, a step-by-step plan, or a quick quiz from this PDF?"
        st.session_state.ui_messages.append(("ai", ack))
        st.session_state.last_citations = []
        st.session_state.last_used_summary_fallback = False
        _render_ui_messages()
        return

    # Special modes before RAG
    if _looks_like_quiz_request(q) and st.session_state.get("text_chunks"):
        quiz = _generate_quiz_from_chunks(st.session_state.text_chunks, load_llm_openai(), n=6)
        st.session_state.ui_messages.append(("ai", quiz))
        st.session_state.last_citations = []
        st.session_state.last_used_summary_fallback = False
        _render_ui_messages()
        return

    if _looks_like_steps_request(q) and st.session_state.get("text_chunks"):
        steps = _generate_steps_from_chunks(st.session_state.text_chunks, load_llm_openai(), n=6)
        st.session_state.ui_messages.append(("ai", steps))
        st.session_state.last_citations = []
        st.session_state.last_used_summary_fallback = False
        _render_ui_messages()
        return

    # RAG
    resp = st.session_state.conversation.invoke({"question": q})
    answer = (resp.get("answer") or "")
    sources = resp.get("source_documents") or []

    # Assess retrieval
    total_chars = sum(len((d.page_content or "").strip()) for d in sources)
    couldnt = "couldn't find" in answer.lower()
    retrieval_empty = (total_chars < 40) or couldnt

    # Auto-summary on weak retrieval for generic/first-touch queries
    use_summary = False
    if retrieval_empty and ( _looks_like_summary_request(q) ):
        if st.session_state.get("text_chunks"):
            with st.spinner("Summarizing the whole PDF..."):
                answer = summarize_all_chunks(st.session_state.text_chunks)
            use_summary = True

    # Append assistant answer to UI history
    st.session_state.ui_messages.append(("ai", answer))
    st.session_state.last_citations = ([] if use_summary else sources)
    st.session_state.last_used_summary_fallback = use_summary

    # Render full chat
    _render_ui_messages()

    # Show citations for the LAST turn only (and only if not a summary fallback)
    if (not use_summary) and sources and total_chars >= 40:
        with st.expander("Citations (retrieved snippets)"):
            for i, d in enumerate(sources, 1):
                preview = (d.page_content or "").strip().replace("\n", " ")
                st.markdown(f"**[{i}]** {preview[:320]}{'…' if len(preview)>320 else ''}")


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    load_dotenv()  # loads OPENAI_API_KEY
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Reset button
    with st.sidebar:
        if st.button("Reset app"):
            st.session_state.clear()
            st.experimental_rerun()

    _ensure_chat_store()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    st.header("Chat with multiple PDFs 📚")

    # Render prior chat (before new input)
    if st.session_state.ui_messages:
        _render_ui_messages()

    # Main question box
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Summarize button
    if st.session_state.get("text_chunks") and st.button("Summarize this PDF"):
        with st.spinner("Summarizing..."):
            summary = summarize_all_chunks(st.session_state.text_chunks)
        # push into UI chat for continuity
        st.session_state.ui_messages.append(("human", "Summarize this PDF"))
        st.session_state.ui_messages.append(("ai", summary))
        st.session_state.last_citations = []
        st.session_state.last_used_summary_fallback = True
        _render_ui_messages()

    # Sidebar upload & processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Test extraction & chunking", disabled=not pdf_docs):
            with st.spinner("Running tests..."):
                test_extract_and_chunk(pdf_docs)

        if st.button("Process", disabled=not pdf_docs):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                total_chars = sum(len(c) for c in text_chunks)
                st.info(f"Indexed {len(text_chunks)} chunks (~{total_chars} characters).")
                if total_chars < 50:
                    st.error(
                        "Very little/no text was indexed. This PDF may be scanned. "
                        "Enable OCR or pre-OCR and try again."
                    )
                    st.stop()

                st.session_state.text_chunks = text_chunks
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Ready! Ask a question above, say 'summarize', 'ask me questions', or 'what are the steps'.")

        # Mini retrieval demo panel
        st.subheader("Mini Retrieval Demo")
        demo_qs_text = st.text_area(
            "Enter 3–5 queries (one per line)",
            value="What is this PDF about?\nList the core concepts.\nGive two key definitions.\nWhere is the method explained?\nSummarize section 1."
        )
        demo_k = st.slider("Show top-k retrieved snippets", 1, 5, 3)
        if st.button("Run retrieval demo", disabled=not st.session_state.get("conversation")):
            with st.spinner("Running mini demo..."):
                run_mini_demo([q for q in demo_qs_text.splitlines() if q.strip()], k=demo_k)

    if not st.session_state.conversation:
        st.info("Upload PDFs and click **Process** to start.")


if __name__ == "__main__":
    main()

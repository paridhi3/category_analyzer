# === Tab 3: File Chatbot ===
# with tab3:
#     st.subheader("💬 Ask Questions About a File")
#     selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

#     if selected_chat_file:
#         file_key = selected_chat_file.replace(".", "_")

#         if f"qa_chain_{file_key}" not in st.session_state:
#             file_text = supported_files[selected_chat_file]
#             vectorstore = create_vector_store(file_text, file_key)
#             qa_chain = get_qa_chain(vectorstore)
#             st.session_state[f"qa_chain_{file_key}"] = qa_chain
#             st.session_state[f"chat_history_{file_key}"] = []

#         qa_chain = st.session_state[f"qa_chain_{file_key}"]
#         chat_history = st.session_state[f"chat_history_{file_key}"]

#         if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
#             chat_history.clear()
#             st.rerun()

#         for msg in chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         user_input = st.chat_input("Type your question:")
#         if user_input:
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             response = qa_chain({"query": user_input})
#             answer = response.get("result", "❌ No answer found.")

#             with st.chat_message("assistant"):
#                 st.markdown(answer)

#             chat_history.append({"role": "user", "content": user_input})
#             chat_history.append({"role": "assistant", "content": answer})
# === Tab 3: File Chatbot ===
with tab3:
    st.subheader("💬 Ask Questions About a File")
    selected_chat_file = st.selectbox("Choose a file for chat:", summary_files, key="chat_file")

    if selected_chat_file:
        file_key = selected_chat_file.replace(".", "_")

        if f"qa_chain_{file_key}" not in st.session_state:
            file_text = supported_files[selected_chat_file]
            vectorstore = load_or_create_vector_store(file_text, file_key)  # ← changed function
            qa_chain = get_qa_chain(vectorstore)
            st.session_state[f"qa_chain_{file_key}"] = qa_chain
            st.session_state[f"chat_history_{file_key}"] = []

        qa_chain = st.session_state[f"qa_chain_{file_key}"]
        chat_history = st.session_state[f"chat_history_{file_key}"]

        if st.button("🔄 Reset Chat", key=f"reset_{file_key}", help="Reset chat history"):
            chat_history.clear()
            st.rerun()

        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your question:")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            response = qa_chain({"query": user_input})
            answer = response.get("result", "❌ No answer found.")

            with st.chat_message("assistant"):
                st.markdown(answer)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})


# === Tab 4: Cross-File Chat ===
# with tab4:
#     st.subheader("📊 Ask Questions Across All Files")
#     if not metadata_cache:
#         st.warning("Please run the categorization first.")
#     else:
#         df = pd.DataFrame(metadata_cache.values())
#         meta_texts = [
#             f"{row['summary']}\nFile: {row['file_name']}\nCategory: {row['category']}\nDomain: {row['domain']}"
#             for _, row in df.iterrows()
#         ]
#         vs = Chroma.from_texts(meta_texts, embedding=embedding_model, collection_name="case_meta", persist_directory=".chromadb_case_meta")
#         cross_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever())
#         cross_query = st.chat_input("Ask a question across all case studies:")
#         if cross_query:
#             with st.spinner("Thinking..."):
#                 result = cross_qa.run(cross_query)
#                 st.markdown(result)
# === Tab 4: Cross-File Chat ===
with tab4:
    st.subheader("📊 Ask Questions Across All Files")
    if not metadata_cache:
        st.warning("Please run the categorization first.")
    else:
        df = pd.DataFrame(metadata_cache.values())
        meta_texts = [
            f"File: {row['file_name']}\nProject Title: {row['project_title']}\nCategory: {row['category']}\nDomain: {row['domain']}\nTechnologies used: {row['technology_used']}\nSummary: {row['summary']}"
            for _, row in df.iterrows()
        ]
        vs = Chroma.from_texts(
            meta_texts,
            embedding=embedding_model,
            collection_name="case_meta",
            persist_directory=".chromadb_case_meta"
        )
        cross_qa = get_cross_file_chain(vs)

        cross_query = st.chat_input("Ask a question across all case studies:")
        if cross_query:
            with st.spinner("Thinking..."):
                result = cross_qa.run(cross_query)
                st.markdown(result)

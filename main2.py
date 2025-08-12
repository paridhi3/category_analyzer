@app.post("/chat")
async def chat_with_metadata(
    query: str = Form(...)
):
    # Read metadata.json content
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_parsed = json.load(f)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or missing metadata JSON file")

    # Step 1: Try metadata-based rule
    meta_answer = answer_from_metadata(query, metadata_parsed)
    if meta_answer:
        return {"response": meta_answer}

    # Step 2: Use Azure OpenAI LLM
    try:
        context_text = json.dumps(metadata_parsed, indent=2)
        prompt = (
            f"Answer the following question using this case study metadata:\n"
            f"{context_text}\n\nQuestion: {query}"
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        bot_reply = response.choices[0].message.content.strip()
        return {"response": bot_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

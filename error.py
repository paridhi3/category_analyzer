chromadb.errors.InvalidArgumentError: Validation error: name: Expected a name containing 3-512 characters from [a-zA-Z0-9._-], starting and ending with a character in [a-zA-Z0-9]. Got: metadata_Case Study 1_pptx

File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 128, in exec_func_with_error_handling
    result = func()
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\main.py", line 286, in <module>
    vectorstore = Chroma.from_documents(
        docs,
    ...<2 lines>...
        persist_directory=None  # In-memory only
    )
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_community\vectorstores\chroma.py", line 887, in from_documents
    return cls.from_texts(
           ~~~~~~~~~~~~~~^
        texts=texts,
        ^^^^^^^^^^^^
    ...<8 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_community\vectorstores\chroma.py", line 817, in from_texts
    chroma_collection = cls(
        collection_name=collection_name,
    ...<5 lines>...
        **kwargs,
    )
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\_api\deprecation.py", line 222, in warn_if_direct_instance
    return wrapped(self, *args, **kwargs)
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_community\vectorstores\chroma.py", line 128, in __init__
    self._collection = self._client.get_or_create_collection(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        name=collection_name,
        ^^^^^^^^^^^^^^^^^^^^^
        embedding_function=None,
        ^^^^^^^^^^^^^^^^^^^^^^^^
        metadata=collection_metadata,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\chromadb\api\client.py", line 232, in get_or_create_collection
    model = self._server.get_or_create_collection(
        name=name,
    ...<3 lines>...
        configuration=configuration,
    )
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\chromadb\api\rust.py", line 268, in get_or_create_collection
    return self.create_collection(
           ~~~~~~~~~~~~~~~~~~~~~~^
        name, configuration, metadata, True, tenant, database
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\chromadb\api\rust.py", line 227, in create_collection
    collection = self.bindings.create_collection(
        name, configuration_json_str, metadata, get_or_create, tenant, database
    )

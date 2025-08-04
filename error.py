ValueError: Missing some input keys: {'query'}

File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 128, in exec_func_with_error_handling
    result = func()
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\main.py", line 284, in <module>
    response = qa_chain({"query": user_input})
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\_api\deprecation.py", line 189, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 410, in __call__
    return self.invoke(
           ~~~~~~~~~~~^
        inputs,
        ^^^^^^^
    ...<2 lines>...
        include_run_info=include_run_info,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 165, in invoke
    self._call(inputs, run_manager=run_manager)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\retrieval_qa\base.py", line 159, in _call
    answer = self.combine_documents_chain.run(
        input_documents=docs,
        question=question,
        callbacks=_run_manager.get_child(),
    )
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\_api\deprecation.py", line 189, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 632, in run
    return self(kwargs, callbacks=callbacks, tags=tags, metadata=metadata)[
           ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\_api\deprecation.py", line 189, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 410, in __call__
    return self.invoke(
           ~~~~~~~~~~~^
        inputs,
        ^^^^^^^
    ...<2 lines>...
        include_run_info=include_run_info,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 163, in invoke
    self._validate_inputs(inputs)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\base.py", line 307, in _validate_inputs
    raise ValueError(msg)

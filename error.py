TypeError: keywords must be strings

File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 128, in exec_func_with_error_handling
    result = func()
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\main.py", line 289, in <module>
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
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\retrieval_qa\base.py", line 156, in _call
    docs = self._get_docs(question, run_manager=_run_manager)
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain\chains\retrieval_qa\base.py", line 278, in _get_docs
    return self.retriever.invoke(
           ~~~~~~~~~~~~~~~~~~~~~^
        question,
        ^^^^^^^^^
        config={"callbacks": run_manager.get_child()},
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\retrievers.py", line 239, in invoke
    **self._get_ls_params(**kwargs),
      ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
File "C:\Users\703417007_agarwal\Desktop\CASE-STUDY-BUILDER\Categorizer\.venv\Lib\site-packages\langchain_core\vectorstores\base.py", line 1057, in _get_ls_params
    ls_params = super()._get_ls_params(**kwargs_)

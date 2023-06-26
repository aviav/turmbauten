# Changelog

All notable changes to this project will be documented in this file.

## [spaghetti-code] - 2023-06-26

This is the first version with _some_ working code that I publish. The
code is yet to be cleaned up a lot, I aspire to add working tests, and
also the README describes some additional functionality that's still
missing. Any interfaces are a temporary snapshot, and I may force-push
to this Git branch at times.

_You need to download and install some stuff before this works._
Details below.

What works?

* Document embedding/vectorization and retrieval

``` shell
python bau.py vectorize --document pride-and-prejudice
```

* Conversation on documents including short-term and long-term memory,
  using llama-based GGML models

    * With document retrieval:
    ``` shell
    python bau.py converse --initial-prompt 'What are pride and prejudice?' \
      --document pride-and-prejudice
    ```

    * Without document retrieval, default initial prompt from config file:
    ``` shell
    python bau.py converse
    ```

For testing, I use Python 3.11.4 and install all packages using `pip
install`. It's possible that more setup is needed. In case something
doesn't work, you're welcome to ask.

For cublas support, I install `llama-cpp-python` as follows:

``` shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python \
  --force-reinstall --upgrade --no-cache-dir
```

In case you don't have a cuda GPU, you can just install
`llama-cpp-python` like any other package using `pip install`, because
then you don't need cublas support. In that case, the models will run
on CPU only, and rather slowly. You can then just remove
`n_gpu_layers=n_gpu_layers` in `bau.py`, and everything should work.

To try it out, have a look at `config.ini` and adjust the paths to
match your system. For use with cuda GPUs, `n_gpu_layers` has to be
adjusted manually so that GPU doesn't run out of memory. You might be
able to free up some GPU RAM by plugging your monitor into a mainboard
port instead of a GPU port.

You need [an embedding
model](https://huggingface.co/spaces/mteb/leaderboard) for document
vectorization and retrieval, and [a LLaMA-based GGML
LLM](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
for conversation.

To create document embeddings from a PDF, you need a PDF. Something
like a book or a yearly report with lots of prose, as opposed to
tables and graphs, should work well. Put it under `DocumentsPath`, in
a subdirectory named like
`pride-and-prejudice/pride-and-prejudice.pdf`.

Depending on the GGML model you can create a model config in
`ModelConfigsPath`. Depending on prompt syntax, you might have to
create three new prompt templates in `TemplatesPath`. It's possible
that you can use one of the existing ones, though.

For now, if you have questions, ask on GitHub issues or LinkedIn. I'll
try to answer as soon as there's time. In case this is all too
complicated for you or runs too slowly to appear worthwhile, you can
achieve very similar things by using e.g. LangChain with OpenAI models
and a cloud-hosted vector database.

What makes my way of using LangChain exciting is that marginal cost is
mainly electric power, while with external APIs you'd often pay per
token, and of course, if everything runs on your own machine instead
of in the cloud, you can even use very sensitive private data without
worrying where it might end up. However, for this, you also need to
have reason to trust in the security of your own machine.

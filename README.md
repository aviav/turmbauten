# Turmbauten

The initial main goal of this project is to let me try out new LLMs
more quickly.

## Getting started

Run `python bau.py`. When I wrote this, this worked for me with Python
3.11.

## Targeted API

Features that I'd like to support, as long as I have to do that:

* Run a multi-feature chain with any suitable new LLM with minimal
  manual config effort. Multi-feature features:
    * Document vectorization
    * Document retrieval
    * Conversation
        * Vector database for long-term memory
        * Short-term memory
    * GGML models in mixed CPU/GPU mode for anything derived from
      LLaMA
        * GGML for Falcon-based models as soon as there's a way to run
          them in mixed mode
    * GPTQ models or better for other LLMs
* Toggles for:
    * Document retrieval
    * Infinite generation mode as seen in llama.cpp
* Sensible defaults for:
    * number of documents retrieved
    * number of long-term memories retrieved
    * short-term memory size

I have many additional ideas for features, but the main challenge is
to limit my goals enough so that there's a working prototype pretty
soon that actually saves me time.

Config:

* Default directory for documents and their vectorizations
* Default directory for conversation long-term memories
* Default directory for machine learning models
* Default directory for prompts
* Path to Llama.cpp which is used for infinite generation mode
* Path to GGML fork that works with Falcon-based models
* Path to Langchain-GPTQ integration

Examples for how to use these features -- _to be implemented, except
where noted otherwise_:

Vectorize 'Pride and Prejudice' -- using Langchain under the hood --
_should already work_:

* Store PDF in `pride-and-prejudice` subdirectory below default
  document directory
* Assuming there's a default embedding model stored in
  `embedding-default` below the default directory for machine learning
  models
* Call `python bau.py vectorize --document pride-and-prejudice`

Start a new conversation without any memory or documents -- using
Langchain under the hood:

* Assuming there's a default LLM with default config stored in
  `llm-default` below the default directory for machine learning
  models
* Assuming the LLM config includes info on context length and
  prompting syntax of the model
* Call `python bau.py converse`

Start a new conversation without any memory, read prompt from file --
using Langchain under the hood:

* Assuming there's a default LLM with default config stored in
  `llm-default` below the default directory for machine learning
  models
* Assuming the LLM config includes info on context length and
  prompting syntax of the model
* Assuming there's a prompt file stored under the given path below the
  default prompts directory
* Call `python bau.py converse --initial-prompt-file test_prompt`

Start a new conversation with document retrieval -- using Langchain
under the hood:

* Assuming Pride and Prejudice has been vectorized
* Assuming there's a sensible default for how many document snippets
  are retrieved on each query
* Assuming the memory of a previous conversation is saved in a
  directory named `thoughts-on-literature` below the long-term
  conversation memory folders
* Assuming there's a sensible default for how many elements of
  long-term memory are retrieved on each query
* Assuming there's a sensible default for short-term memory context
  length
* Assuming there's a default LLM with default config stored in
  `llm-default` below the default directory for machine learning
  models
* Assuming the LLM config includes info on context length and
  prompting syntax of the model
* Call `python bau.py converse --doc-db pride-and-prejudice
  --memory-db thoughts-on-literature --short-term-memory
  --initial-prompt 'Why is there pride and prejudice?`

Start a new conversation in infinite mode -- using Llama.cpp or a fork
supporting Falcon under the hood -- without any kind of memory or
document retrieval:

* Assuming there's a default LLM with default config stored in
  `llm-default` below the default directory for machine learning
  models
* Assuming the LLM config includes info on context length, llama-based
  vs. falcon-based and prompting syntax of the model
* Call `python bau.py converse --infinite --initial-prompt 'How're you
  doing today?'`

## Contributing

My favorite kind of external contribution is if you start to maintain
a fork that respects the license, and if that fork then becomes
popular. In case I end up supporting this project, I'll pull in
changes that I find useful for my personal usage.

If you find the license too restrictive, start a project that uses
your own code and documentation for the same purpose, and
publish/license it as needed. For stuff that I write myself, I prefer
a license that actively promotes and enforces freedom as an end. For
spreading the underlying idea, people are welcome to use their own
skills and their own license.

My main reason for publishing this repo is to spread the news that
what I describe is possible -- I have done all of the targeted things
in an unorganized fashion on my dev machine already -- and has a low
threshold to entry. As soon as the knowledge about what's possible is
common enough, I'm sure there'll be better maintainers than myself.

## Acknowledgments

Various LLMs and open-source programs have been used to create this
project. This was important because I don't really know how to Python.

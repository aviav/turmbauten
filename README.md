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

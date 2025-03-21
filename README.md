### Simple LLM Training and Generation Application

A simple application for training LLM and generate text using a similar architecture to GPT with references from "LLM from Scratch" by Sebastian Raschka. The application can be executed by running the command line `python gpt_app.py`. `python3.11` version is recommended.

`gpt_app.py`

Main program

`gpt_model.py`

Contains classes that made up the architecture for training LLM and generating text

`gpt_app_ui.py`

Alternative program for text generation with UI. Model and Config names must be "model.pth" and "model.yaml" respectively to work. Execute by running `chainlit run gpt_app_ui.py`.

`sample\biology.pth (not in repository)` 

Sample model trained from `biology.txt` with default configurations. Can be obtained from https://github.com/finaea/llm_training_generation/releases/tag/sample

`sample\biology.yaml`

Default configurations used for `biology.pth`. The program reads only yaml file.

`sample\biology.txt`

Raw text extracted directly from Chapter 1 in Cambridge International AS & A Level Biology Fourth Edition

`sample\loss.pdf`

Line Graph that is generated after training. Shows the training and validation progress over epochs

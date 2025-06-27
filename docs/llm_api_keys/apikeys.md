# LLM APIs

Empirica requires access to large language models (LLMs) to function. Currently, Empirica supports LLMs from [Google (Gemini series)](https://ai.google.dev/gemini-api/docs/models?hl=es-419), [OpenAI (GPT and o series)](https://platform.openai.com/docs/models), [Anthropic (Claude)](https://www.anthropic.com/claude), [Perplexity (Sonar)](https://sonar.perplexity.ai/), and agents from [Futurehouse (Owl)](https://platform.futurehouse.org/). Access to all these models is not mandatory for experimentation; however, **at least OpenAI API access is required for the Analysis module**, so an OpenAI API key must be configured if that module is expected to be employed.

API access is managed via keys generated on each provider's platform and set as environment variables. Most LLM providers require a small amount of credit to be added to your account, as usage typically incurs a cost (though this is relatively minor for experimentation).

The table below summarizes which LLM models are required ([REQUIRED]), optional ([OPTIONAL]) or not employed ([NOT USED]) for each of the Empirica modules:

| Module             | OpenAI | Gemini | Vertex AI | Claude | Perplexity | FutureHouse |
| ------------------ | ------ | ------ | --------- | ------ | ---------- | ----------- |
| **Generate Ideas** | [OPTIONAL]     | [OPTIONAL]     | [OPTIONAL]        | [OPTIONAL]     | [NOT USED]         | [NOT USED]          |
| **Methods**        | [OPTIONAL]     | [OPTIONAL]     | [OPTIONAL]        | [OPTIONAL]     | [NOT USED]         | [NOT USED]          |
| **Analysis**       | [REQUIRED]     | [OPTIONAL]     | [OPTIONAL]        | [OPTIONAL]     | [NOT USED]         | [NOT USED]          |
| **Paper Writing**  | [OPTIONAL]     | [OPTIONAL]     | [NOT USED]        | [OPTIONAL]     | [NOT USED]         | [NOT USED]          |
| **Paper Review**   | [OPTIONAL]     | [OPTIONAL]     | [NOT USED]        | [OPTIONAL]     | [NOT USED]         | [NOT USED]          |
| Citation Search    | [NOT USED]     | [NOT USED]     | [NOT USED]        | [NOT USED]     | [REQUIRED]         | [NOT USED]          |
| Check Idea         | [OPTIONAL]     | [OPTIONAL]     | [OPTIONAL]        | [OPTIONAL]     | [OPTIONAL]         | [OPTIONAL]          |

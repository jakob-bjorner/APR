Answering questions I had while reading the Adaptive Parallel Reasoning paper.

What is the inference speed up of SGLang relative to huggingface decoding? (code)
What do their SGLang batching optimizations look like in code? (code)
How do they batch this or how does SGLang handle multitoken inference requests when some batch elements are shorter? (code)
Do they incorporate time into their reward function? Or just accuracy? (code)

## getting started on HPC-AI.com compute.
They have a nice jupyter lab environment, and if you set up permanent sotrage its like $1/day, and per hour is $15 when using h100 node of 8. Cant go smaller tho. For modal, this would be best, but unfamiliar with it. Just waiting till I have access to compute with 80 gb vram.
# Phi-Fibonacci Compression
The goal of this project is to create a novel compression algorithm that works on all data types.

For the sample-2.wave file:

Chunk size: 128-bit ()
Total chunks: 10728970 
Estimated size: 42,915,880 bytes

So here's the idea:

We want to analyze the bit-string squences within a file and then use a shorthand to describe them.

We may vary the length of the window. What utilities can do this automatically? Markov analysis, grep engines? Huffman encoding? Etc?

For the common patterns at the first level, we create a dictionary. We create a representation of those values with a pattern-lookup. We then recursively apply the process.

To clarify:

If we have 42915880 chunks into an 8-bit pattern, we have 256 patterns, all possible values for an 8-bit mapping. Nothing falls outside of this, and we leave the problem of padding for later.
```
42,915,880 total occurrences
```
Now we examine a 16-bit window:

We have 21457940 chunks into 16-bit patterns, we have:
```
Pattern count analysis:
Patterns occurring >2 times: 34498 patterns, 21455681 total occurrences
Patterns occurring =2 times: 557 patterns, 1114 total occurrences
Patterns occurring =1 time:  1145 patterns, 1145 total occurrences
```

So now we have leftovers, these will live in their own dictionary (1-offs, 2-offs).

Then we look at the 34498 patterns with 24-bit:
```
Patterns occurring >2 times: 1431906 patterns, 10403509 total occurrences
Patterns occurring =2 times: 983375 patterns, 1966750 total occurrences
Patterns occurring =1 time:  1935034 patterns, 1935034 total occurrences
Total chunks: 14305293
```

Now, to do this right, we should be 
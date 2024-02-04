# Tests

I'll be using this test question for all tests because it's complicated enough that it requires accessing a few docs.

Note that the smart chunking algo is being compaired to LangChain's RecursiveTextChunkSplitter with sizes of 512
## Equal k Test
Test both with k=60 to ensure that they both can get ample context. See that they both perform well, and/or the smart chunking algo performs slightly better with vastly fewer context tokens required

## Equal Token Test
Leaving the smart chunker at 60, set the k of the default algo to 10-20 (it won't be exact) and see that the smart chunking algo performs MUCH better
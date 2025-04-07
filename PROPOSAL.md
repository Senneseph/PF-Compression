Look at webcamp.py to see what I was doing.

While leaving that file alone and where it is, we want to turn the contents of the file into a sensible directory structure.

I have Transformers, Encoders, Decoders, and Filters. These are all high-level abstract patterns. Look for the similiary-name functions and try to break them apart into classes, interfaces, or whatever abstraction pattern suits.

The eventual goal is to create a library, hosted on git that provides these effects.
In another project, we will use this library to create a web interface to the effects and use the web browser to ask permission for audio and video. That folder is in the parent directory and named "Video Effects".
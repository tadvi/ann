ann - Artificial Neural Networks in Go
======================================

Go (golang) implementations of various Neural Networks. 

* som.go is simple implementation of Self-Organizing Maps also known as Kohonen's maps.
* backprop.go is backpropagation training based neural network.

Check out demo.go for few examples on how networks can be used.

Examples
========

Examples in demo.go are too simplistic for use in real life. They just show 
how neural networks can be used. Some of examples, like prime number prediction, 
can be solved with simple look-up table. 

Same idea applies to number of problems in the real world - if we would have all possible 
examples of input data we could put them into big look-up table and there would be 
no need for neural networks at all.
Generalisation makes neural networks useful. They can produce sensible outputs
for inputs that weren't encoutered during learning.



## Development Diaries

I am using this file to basically log the process I use to solve this problem as it has been 2 days and I have not been able to solve it yet.

### 18/04/2024

The forward pass has been completed and was really simple, nothing major to note except for some shape shifting shenanigans.

The part that is still left is backward pass. Here is what I have been able to do till now - 

- Let us have a neural network with 3 layers (excluding the input layer), $L_1$, $L_2$, $L_3$.
![Base Neural Net](./nn.svg "Neural Network")
- For input data, X, let - 
  - $n_s$ = number of examples
  - $n_f$ = number of features
- For $L_i$, let - 
  - $n_i$ = number of neurons in the layer
  - $W_i$ = weights of the layer, shape = ($n_{i - 1}$, $n_i$)
  - $b_i$ = biases of the layer, shape = ($n_s$, $n_i$)
  - $a_i$ = output of the layer (after applying activation function, relu for this project), shape = ($n_s$, $n_i$)

- The forward pass is simple enough. For the backward pass, after a lot of thinking, I have though of a method although it is not working yet.
  - For example, to calculate $\frac{dJ}{dW_3}$, what we can do is first calculate $\frac{dJ}{da_3}$, then calculate $\frac{da_3}{dW_3}$. After that we can use chain rule.
  $$\frac{dJ}{dW_3} = \frac{dJ}{da_3} \cdot \frac{da_3}{dW_3}$$
  $$\frac{dJ}{dW_3} = \frac{d}{da_3}(\frac{(a_3 - y)^2}{2m}) \cdot \frac{d}{dW_3}(\sigma(a_2 \cdot W_3 + b_3))$$ 
  $$\frac{dJ}{dW_3} = \frac{(a_3 - y)}{m} \cdot \sigma\rq(a_2 \cdot W_3 + b_3) \cdot a_2$$ 

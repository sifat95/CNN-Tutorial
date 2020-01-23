# Convolutional Neural Network

# Building Blocks of a Neural Network using Convolution

![Convolutional Layer](https://github.com/sifat95/CNN-Tutorial/blob/master/images/model.png)

# Code Implementation
### Convolution Functions
 1. Zero Padding
 2. Convolve window
 3. Convolution forward
 4. Convolution backward
### Pooling Functions 
 1. Pooling forward
 2. Create mask
 3. Distribute value
 4. Pooling backward

## Convolution Functions
### Two helper functions

 1. **Zero Padding**: As we know most of the time deeper networks are built which may shrink the height and width of the volumes causing data loss. So zero padding adds zeros around an image that helps us keep more of the information at the border of that image.
 
![padding](https://github.com/sifat95/CNN-Tutorial/blob/master/images/PAD.png)

Now let's pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, as illustrated in Figure 1.

#### Arguments of the Function:
X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
pad -- integer, amount of padding around each image on vertical and horizontal dimensions

#### The Function Returns:
X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)   

    def zero_pad(X, pad):

        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
       
        return X_pad

2. **Single Step of Convolution**: A convolutional unit is that which takes an input volume and applies a filter at every position of the input to produce a new volume. So in the convolutional unit, many single step convolutions are performed i.e. a filter is applied to every single position of the input that eventually builds a convolutional unit.

![convolution](https://github.com/sifat95/CNN-Tutorial/blob/master/images/Convolution_schematic.gif)

Below is the code for single step convolution. It applies one filter defined by parameters W on a single slice (a_slice_prev) of the output activation of the previous layer to get a single real-valued output. Later, this function will be applied to multiple positions of the input to implement the full convolutional operation.

#### Arguments of the Function:
a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
#### The Function Returns:
Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data

    def conv_single_step(a_slice_prev, W, b):

        # Element-wise product between a_slice_prev and W. Do not add the bias yet.
        s = np.multiply(a_slice_prev, W)
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z + float(b)

        return Z
        
### Convolution
 1. **Forward Propagation**
Implements the forward propagation for a convolution function
#### Arguments of the Function:
A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
b -- Biases, numpy array of shape (1, 1, 1, n_C)
hparameters -- python dictionary containing "stride" and "pad"
        
#### The Function Returns:
Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
cache -- cache of values needed for the conv_backward() function

    def conv_forward(A_prev, W, b, hparameters):
    
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters['stride']
        pad = hparameters['pad']

        # Compute the dimensions of the CONV output volume using the formula given above. 
        # Hint: use int() to apply the 'floor' operation. (≈2 lines)
        n_H = int((n_H_prev - f + (2 * pad)) / stride + 1)
        n_W = int((n_W_prev - f + (2 * pad)) / stride + 1)

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = zero_pad(A_prev, pad)

        for i in range(m):                               # loop over the batch of training examples
            a_prev_pad = A_prev[i]                           # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = stride * h
                        vert_end = stride * h + f
                        horiz_start = stride * w
                        horiz_end = stride * w + f

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))

        # Save information in "cache" for the backprop
        cache = (A_prev, W, b, hparameters)

        return Z, cache
        
  2. **Backward Propagation**
 
 Implement the backward propagation for a convolution function
    
#### Arguments of the Function:
dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
cache -- cache of values needed for the conv_backward(), output of conv_forward()
#### The Function Returns:
dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
 
    def conv_backward(dZ, cache):

        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters"
        stride = hparameters['stride']
        pad = hparameters['pad']

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.random.randn(m, n_H_prev, n_W_prev, n_C_prev)
        dW = np.random.randn(f, f, n_C_prev, n_C)
        db = np.random.randn(f, f, n_C_prev, n_C)

        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        for i in range(m):                       # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]

            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = stride * h
                        vert_end = stride * h + f
                        horiz_start = stride * w
                        horiz_end = stride * w + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, dW, db

 

## Pooling Functions
### Forward Pass of the Pooling Layer
Implements the forward pass of the pooling layer
    
#### Arguments of the Function:
A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
hparameters -- python dictionary containing "f" and "stride"
mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
#### The Function Returns:
A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 

    def pool_forward(A_prev, hparameters, mode = "max"):
    
       # Retrieve dimensions from the input shape
       (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

       # Retrieve hyperparameters from "hparameters"
       f = hparameters["f"]
       stride = hparameters["stride"]

       # Define the dimensions of the output
       n_H = int(1 + (n_H_prev - f) / stride)
       n_W = int(1 + (n_W_prev - f) / stride)
       n_C = n_C_prev

       # Initialize output matrix A
       A = np.zeros((m, n_H, n_W, n_C))              

       for i in range(m):                         # loop over the training examples
           for h in range(n_H):                      # loop on the vertical axis of the output volume
               for w in range(n_W):                 # loop on the horizontal axis of the output volume
                   for c in range (n_C):            # loop over the channels of the output volume

                       # Find the corners of the current "slice" (≈4 lines)
                       vert_start = stride * w
                       vert_end = stride * w + f
                       horiz_start = stride * h
                       horiz_end = stride * h + f

                       # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                       a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                       # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                       if mode == "max":
                           A[i, h, w, c] = np.max(a_prev_slice)
                       elif mode == "average":
                           A[i, h, w, c] = np.mean(a_prev_slice)

       # Store the input and hparameters in "cache" for pool_backward()
       cache = (A_prev, hparameters)

       # Making sure your output  shape is correct
       assert(A.shape == (m, n_H, n_W, n_C))

       return A, cache

### Backward Pass of the Pooling layer

 1. **Creating Mask**
Creates a mask from an input matrix x, to identify the max entry of x.
    
#### Arguments of the Function:
x -- Array of shape (f, f)
    
#### The function Returns:
mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.

    def create_mask_from_window(x):
    
        mask = (x == np.max(x))
    
        return mask
        
 2. **Distribute the Value**
Distributes the input value in the matrix of dimension shape
    
#### Arguments of the Function:
dz -- input scalar
shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
#### The Function Returns:
a -- Array of size (n_H, n_W) for which we distributed the value of dz

    def distribute_value(dz, shape):

        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape

        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)

        # Create a matrix where every entry is the "average" value (≈1 line)
        a = average * np.ones(shape)

        return a
        
 3. **Backward Pass**
Implements the backward pass of the pooling layer
    
#### Arguments of the Function:
dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
#### The Function Returns:
dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    
    def pool_backward(dA, cache, mode = "max"):

        # Retrieve information from cache (≈1 line)
        (A_prev, hparameters) = cache

        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = hparameters['stride']
        f = hparameters['f']

        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = None

        for i in range(m):                       # loop over the training examples

            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i, :, :, :]

            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = stride * h
                        vert_end = stride * h + f
                        horiz_start = stride * w
                        horiz_end = stride * w + f

                        # Compute the backward propagation in both modes.
                        if mode == "max":

                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[horiz_start:horiz_end, vert_start:vert_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += a_prev_slice * mask

                        elif mode == "average":

                            # Get the value a from dA (≈1 line)
                            da = np.sum(a_prev[horiz_start:horiz_end, vert_start:vert_end, c])
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] +=  distribute_value(da, shape)

        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)

        return dA_prev

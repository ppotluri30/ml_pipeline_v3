import torch
import torch.nn as nn
import math
import random
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size: int, n_exo_features: int, hidden_size: int, output_size: int, num_layers: int=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.n_endo_features = input_size - n_exo_features
        self.n_exo_features = n_exo_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * self.n_endo_features)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: The output tensor from the model.
        """
        # --- Initialize Hidden and Cell States ---
        # The LSTM needs initial hidden and cell states. If not provided, they default to zeros.
        # The shape is (num_layers, batch_size, hidden_size).
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # --- LSTM Forward Pass ---
        # The LSTM returns the output of the entire sequence and the final hidden and cell states.
        # out shape: (batch_size, sequence_length, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        # cn shape: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # --- Fully-Connected Layer ---
        # We are interested in the output of the last time step for many sequence tasks
        # (e.g., classification, next value prediction).
        # out[:, -1, :] selects the output of the last element in the sequence for each batch.
        # Shape of out[:, -1, :]: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        
        # Reshape the output to be (batch_size, output_seq_len, output_features)
        # This matches the shape of our target tensor.
        out = out.view(x.size(0), self.output_size, self.n_endo_features)
        
        return out

# WIP model that would allow for scheduled sampling
class EncoderLSTM(nn.Module):

    class Encoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Processes the input sequence and returns the final hidden and cell states.
            """
            # The outputs of the LSTM are ignored here; we only need the context.
            _, (hidden, cell) = self.lstm(x)
            return hidden, cell
        
    class Decoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
            super().__init__()
            # The decoder's input size is all features (endogenous + exogenous)
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            # The output layer predicts only the endogenous (target) features
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x_step: torch.Tensor,
                    hidden: torch.Tensor,
                    cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Performs a single decoding step.
            """
            # x_step needs to be reshaped to (batch_size, 1, num_features) for the LSTM
            x_step = x_step.unsqueeze(1)

            # Process one time step
            output, (hidden, cell) = self.lstm(x_step, (hidden, cell))

            # Pass the LSTM output through the fully connected layer
            prediction = self.fc(output.squeeze(1))

            return prediction, hidden, cell
        
    def __init__(self, input_size: int, n_exo_features: int, hidden_size: int, output_seq_len: int, num_layers: int = 1):
        super(EncoderLSTM, self).__init__()
        self.n_endo_features = input_size - n_exo_features
        self.n_exo_features = n_exo_features
        self.output_seq_len = output_seq_len
        
        # The decoder input includes all features
        decoder_input_size = self.n_endo_features + self.n_exo_features
        
        # The decoder output is just the endogenous (target) features
        decoder_output_size = self.n_endo_features
        
        self.encoder = self.Encoder(input_size, hidden_size, num_layers)
        self.decoder = self.Decoder(decoder_input_size, hidden_size, decoder_output_size, num_layers)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        The main forward pass that orchestrates the encoder-decoder architecture.
        
        Args:
            src (torch.Tensor): The source/input sequence. Shape: (batch, input_len, features).
            trg (torch.Tensor): The target sequence. Shape: (batch, output_len, features).
                                This contains the ground-truth targets and the known future exogenous features.
            teacher_forcing_ratio (float): The probability of using the ground truth for the next input.
        """
        batch_size = src.shape[0]
        device = src.device

        # A tensor to store the decoder's predictions
        outputs = torch.zeros(batch_size, self.output_seq_len, self.n_endo_features).to(device)

        # 1. Encode the source sequence to get the context
        hidden, cell = self.encoder(src)

        # 2. Prepare the first input for the decoder
        # This will be the last time step of the source sequence
        decoder_input = src[:, -1, :]

        # 3. Decode step-by-step
        for t in range(self.output_seq_len):
            # Generate a prediction for the current time step
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # Store the prediction
            outputs[:, t] = prediction
            
            # Decide whether to use teacher forcing for the next time step
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Use the ground truth as the next input
                # We take the known endogenous features from the target tensor 'trg'
                endo_features = trg[:, t, :self.n_endo_features]
            else:
                # Use the model's own prediction as the next endogenous input
                endo_features = prediction
            
            if self.n_exo_features > 0:
                # We always use the known future exogenous features from the 'trg' tensor
                exo_features = trg[:, t, self.n_endo_features:]
                decoder_input = torch.cat([endo_features, exo_features], dim=1)
            else:
                decoder_input = endo_features
                
        return outputs
    def forecast(self, src: torch.Tensor, future_exo: torch.Tensor) -> torch.Tensor:
        """
        Forecasts future endogenous values given past data and known future exogenous features.

        Args:
            src (torch.Tensor): Source sequence, shape (batch, input_len, features).
            future_exo (torch.Tensor): Known future exogenous features,
                                       shape (batch, output_len, n_exo_features).
        Returns:
            torch.Tensor: Forecasted endogenous values,
                          shape (batch, output_len, n_endo_features).
        """
        batch_size = src.shape[0]
        device = src.device

        outputs = torch.zeros(batch_size, self.output_seq_len, self.n_endo_features, device=device)

        hidden, cell = self.encoder(src)
        decoder_input = src[:, -1, :]

        for t in range(self.output_seq_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = prediction

            if self.n_exo_features > 0:
                exo_features = future_exo[:, t, :]
                decoder_input = torch.cat([prediction, exo_features], dim=1)
            else:
                decoder_input = prediction

        return outputs

    def predict(self, src_np: np.ndarray, future_exo_np: np.ndarray | None = None) -> np.ndarray:
        """
        NumPy-friendly wrapper for forecast(), matching the eval loop's expectations.

        Args:
            src_np (np.ndarray): Source sequence, shape (batch, input_len, features).
            future_exo_np (np.ndarray, optional): Known future exogenous features,
                                                  shape (batch, output_len, n_exo_features).
                                                  If None, assumes no exogenous.

        Returns:
            np.ndarray: Forecasted endogenous values,
                        shape (batch, output_len, n_endo_features).
        """
        device = next(self.parameters()).device
        src = torch.from_numpy(src_np).float().to(device)

        if self.n_exo_features > 0:
            if future_exo_np is None:
                raise ValueError("future_exo_np must be provided when model has exogenous features")
            future_exo = torch.from_numpy(future_exo_np).float().to(device)
        else:
            future_exo = torch.zeros(src.shape[0], self.output_seq_len, 0, device=device)

        with torch.no_grad():
            preds = self.forecast(src, future_exo)
        return preds.cpu().numpy()

    

class GRU(nn.Module):
    def __init__(self, input_size: int, n_exo_features: int, hidden_size: int, output_size: int, num_layers: int=1):
        """
        Initializes the GRUModel.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers.
            output_size (int): The number of features in the output layer.
        """
        super(GRU, self).__init__() # Call the constructor of the parent class (nn.Module)
        self.input_size = input_size
        self.n_endo_features = input_size - n_exo_features
        self.n_exo_features = n_exo_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Define the GRU layer
        # batch_first=True means the input and output tensors are provided as (batch, seq, feature)
        # instead of (seq, batch, feature). This is generally more intuitive.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layer that maps the hidden state to the desired output size
        self.fc = nn.Linear(hidden_size, output_size * self.n_endo_features)

    def forward(self, x):
        """
        Defines the forward pass of the GRU model.

        Args:
            x (torch.Tensor): The input tensor. Expected shape: (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor. Expected shape: (batch_size, output_size).
        """
        # Initialize the hidden state with zeros
        # The hidden state has shape (num_layers * num_directions, batch_size, hidden_size).
        # For a simple GRU, num_directions is 1.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the GRU layer
        # output: tensor containing the output features (h_t) from the last layer of the GRU,
        #         for each t. Shape: (batch_size, sequence_length, hidden_size)
        # hn: tensor containing the hidden state for the last timestep of each layer.
        #     Shape: (num_layers * num_directions, batch_size, hidden_size)
        output, hn = self.gru(x, h0)

        # We are interested in the output from the last timestep of the last layer.
        # Since batch_first=True, output[:, -1, :] gives the output of the last timestep
        # for all sequences in the batch.
        # Then, pass this through the fully connected layer.
        out = self.fc(output[:, -1, :])

        out = out.view(x.size(0), self.output_size, self.n_endo_features)

        return out
    

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for adding positional information to the input embeddings.
    Required because Transformers are permutation-invariant without it.
    """
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension: (1, max_len, model_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, model_dim).
        """
        # Add positional encoding to the input.
        # Ensure sequence_length of x does not exceed max_len
        x = x + self.pe[:, :x.size(1)] # type: ignore
        return x

class TETS(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_exo_features: int, model_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 3, feedforward_dim: int = 512, dropout: float = 0.1):
        """
        Initializes the Transformer Encoder model for time series.

        Args:
            input_size (int): The number of features in each time step of the input.
            output_size (int): The length of the output sequence to be predicted.
            n_exo_features (int): The number of exogenous features.
            model_dim (int): The dimension of the embeddings (and attention).
            num_heads (int): The number of attention heads.
            num_layers (int): The number of stacked encoder layers.
            feedforward_dim (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TETS, self).__init__()
        self.output_size = output_size
        self.n_endo_features = input_size - n_exo_features
        self.model_dim = model_dim

        # Linear layer to project input features to model_dim dimension
        self.input_projection = nn.Linear(input_size, model_dim)
        
        # Positional Encoding to inject information about the position of each element in the sequence
        self.positional_encoding = PositionalEncoding(model_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                   dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final fully connected layer to map the Transformer's output to the desired prediction
        # We take the output of the last token (time step) from the Transformer.
        self.fc = nn.Linear(model_dim, output_size * self.n_endo_features)

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the Transformer Encoder model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        # Project input features to model_dim dimension
        # Shape: (batch_size, sequence_length, model_dim)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through Transformer Encoder
        # Shape: (batch_size, sequence_length, model_dim)
        transformer_output = self.transformer_encoder(x)

        # For time series forecasting, we often take the output of the last sequence element
        # as the representation from which to make the prediction.
        # Shape: (batch_size, model_dim)
        last_time_step_output = transformer_output[:, -1, :]

        # Pass through the final fully connected layer
        # Shape: (batch_size, output_size * n_endo_features)
        out = self.fc(last_time_step_output)

        # Reshape the output to be (batch_size, output_size, n_endo_features)
        out = out.view(x.size(0), self.output_size, self.n_endo_features)

        return out
    

class Chomp1d(nn.Module):
    """
    To remove the padding introduced by causal convolution (padding applied only on the left).
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A block in the Temporal Convolutional Network.
    Consists of two dilated causal convolutional layers, followed by
    weight normalization, ReLU, and dropout. Includes a residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # Removes the padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection: if input and output feature dimensions differ, apply a 1x1 convolution
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # Apply residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_exo_features: int, 
                 layer_architecture: list, kernel_size: int = 2, dropout: float = 0.2):
        """
        Initializes the Temporal Convolutional Network model.

        Args:
            input_size (int): The number of features in each time step of the input.
            output_size (int): The length of the output sequence to be predicted.
            n_endo_features (int): The number of endogenous features to predict.
            layer_architecture (list): A list defining the number of output channels for each TemporalBlock.
            kernel_size (int): The size of the convolutional kernel for each block.
            dropout (float): The dropout value.
        """
        super(TCN, self).__init__()
        self.output_size = output_size
        self.n_endo_features = input_size - n_exo_features
        
        layers = []
        num_levels = len(layer_architecture)
        for i in range(num_levels):
            dilation_size = 2 ** i # Exponentially increasing dilation
            in_channels = input_size if i == 0 else layer_architecture[i-1]
            out_channels = layer_architecture[i]
            # Padding to ensure the output sequence length remains the same after causal convolution
            # padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.tcn_layers = nn.Sequential(*layers)

        # Final fully connected layer to map the TCN's output to the desired prediction
        # We take the output of the last time step from the TCN.
        self.fc = nn.Linear(layer_architecture[-1], output_size * self.n_endo_features)

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        # TCN expects input of shape (batch_size, features, sequence_length)
        # Our input is (batch_size, sequence_length, input_size), so we need to permute.
        x = x.permute(0, 2, 1) # Now x is (batch_size, input_size, sequence_length)

        # Pass through TCN layers
        # Shape: (batch_size, layer_architecture[-1], sequence_length)
        tcn_out = self.tcn_layers(x)

        # For time series forecasting, we typically take the output of the last time step
        # (which corresponds to the latest available information).
        # Shape: (batch_size, layer_architecture[-1])
        last_time_step_output = tcn_out[:, :, -1]

        # Pass through the final fully connected layer
        # Shape: (batch_size, output_size * n_endo_features)
        out = self.fc(last_time_step_output)

        # Reshape the output to be (batch_size, output_size, n_endo_features)
        out = out.view(x.size(0), self.output_size, self.n_endo_features)

        return out
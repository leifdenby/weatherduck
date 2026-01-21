import fiddle
import fiddle.experimental
import fiddle.experimental.auto_config
from torch import nn
from torch_geometric.nn import SAGEConv

from ...step_predictor import (
    EncodeProcessDecodeModel,
    Processor,
    SingleNodesetDecoder,
    SingleNodesetEncoder,
    TrainableFeatureManager,
    make_mlp,
)

__all__ = ["build_encode_process_decode_model"]


@fiddle.experimental.auto_config.auto_config
def build_encode_process_decode_model(
    *,
    n_input_data_features: int,
    n_output_data_features: int,
    n_hidden_data_features: int,
    n_input_trainable_features: int,
    n_hidden_trainable_features: int,
    hidden_dim: int,
) -> EncodeProcessDecodeModel:
    """Build an Encode-Process-Decode model with SAGEConv components.

    Parameters:
        n_input_data_features: Number of input data features for each node.
        n_output_data_features: Number of output data features to predict.
        n_hidden_data_features: Number of hidden data features per node.
        n_input_trainable_features: Number of trainable input features per node.
        n_hidden_trainable_features: Number of hidden trainable features per node.
        hidden_dim: Hidden dimension used across encoder/processor/decoder.

    Returns:
        EncodeProcessDecodeModel configured with the requested dimensions.
    """
    encoder = SingleNodesetEncoder(
        embedder_src=make_mlp(
            n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim
        ),
        embedder_dst=make_mlp(
            n_hidden_data_features + n_hidden_trainable_features, hidden_dim, hidden_dim
        ),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=nn.Linear(hidden_dim, hidden_dim),
    )
    processor = Processor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = SingleNodesetDecoder(
        embedder_src=make_mlp(
            hidden_dim + n_hidden_data_features + n_hidden_trainable_features,
            hidden_dim,
            hidden_dim,
        ),
        embedder_dst=make_mlp(
            n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim
        ),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=nn.Linear(hidden_dim, n_output_data_features),
    )

    trainable_manager = TrainableFeatureManager(
        n_input_trainable_features, n_hidden_trainable_features
    )

    return EncodeProcessDecodeModel(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        trainable_manager=trainable_manager,
    )

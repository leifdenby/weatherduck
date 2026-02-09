import inspect
from typing import Optional

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing

__all__ = [
    "TwoLayerMLP",
    "TrainableFeatures",
    "TrainableFeatureManager",
    "run_message_op",
    "SingleNodesetEncoder",
    "Processor",
    "SingleNodesetDecoder",
    "EncodeProcessDecodeModel",
]


class TwoLayerMLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden layer size.
        out_dim : int
            Output feature dimension.

        Returns
        -------
        None
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.net(x)


class TrainableFeatures(nn.Module):
    """
    Trainable node features stored as a parameter and concatenated at runtime.

    Parameters
    ----------
    num_nodes : int
        Number of nodes these features correspond to.
    n_features : int
        Feature length of the learnable vector per node.

    Returns
    -------
    torch.Tensor
        Trainable features of shape [num_nodes, n_features].
    """

    def __init__(self, num_nodes: int, n_features: int):
        """Initialize trainable feature parameters.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the template graph.
        n_features : int
            Trainable feature dimension per node.

        Returns
        -------
        None
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.n_features = n_features
        param = nn.Parameter(torch.zeros(num_nodes, n_features))
        nn.init.constant_(param, 0.0)
        self.register_parameter("trainable", param)

    def forward(self, current_num_nodes: int) -> torch.Tensor:
        """Return trainable features for a (possibly batched) graph.

        Parameters
        ----------
        current_num_nodes : int
            Number of nodes in the current graph batch.

        Returns
        -------
        torch.Tensor
            Trainable features aligned with nodes.
        """
        # Supports batched HeteroData where num_nodes can be a multiple of the
        # template node count. If so, repeat the trainable features per graph.
        if current_num_nodes == self.num_nodes:
            return self.trainable
        if current_num_nodes % self.num_nodes == 0:
            repeat = current_num_nodes // self.num_nodes
            return self.trainable.repeat(repeat, 1)
        raise AssertionError(
            f"Trainable features expect {self.num_nodes} or a multiple thereof, got {current_num_nodes}"
        )


class TrainableFeatureManager(nn.Module):
    """
    Manages per-graph trainable features for data and hidden nodes.

    Parameters
    ----------
    n_input_trainable_features : int
        Trainable feature length per data node.
    n_hidden_trainable_features : int
        Trainable feature length per hidden node.
    """

    def __init__(
        self, n_input_trainable_features: int, n_hidden_trainable_features: int
    ):
        """Initialize the trainable feature manager.

        Parameters
        ----------
        n_input_trainable_features : int
            Trainable feature dimension for data nodes.
        n_hidden_trainable_features : int
            Trainable feature dimension for hidden nodes.

        Returns
        -------
        None
        """
        super().__init__()
        self.n_input_trainable_features = n_input_trainable_features
        self.n_hidden_trainable_features = n_hidden_trainable_features
        self.data_modules = nn.ModuleDict()
        self.hidden_modules = nn.ModuleDict()

    def build_features(
        self, graph: HeteroData, node_type: str
    ) -> Optional[torch.Tensor]:
        """
        Build trainable features for a node type in a (possibly batched) graph.

        Parameters
        ----------
        graph : HeteroData
            Batched or single graph containing graph_id and batch vectors.
        node_type : str
            'data' or 'hidden'.

        Returns
        -------
        torch.Tensor | None
            Trainable features aligned with nodes of the given type, or None if
            the requested feature dimension is 0.
        """
        feature_dim = (
            self.n_input_trainable_features
            if node_type == "data"
            else self.n_hidden_trainable_features
        )
        if feature_dim == 0:
            return None
        device = graph[node_type].x.device
        modules = self.data_modules if node_type == "data" else self.hidden_modules
        batch_vec = (
            graph[node_type].batch
            if "batch" in graph[node_type]
            else torch.zeros(
                graph[node_type].num_nodes, dtype=torch.long, device=device
            )
        )
        num_graphs = (
            graph.num_graphs
            if hasattr(graph, "num_graphs")
            else (int(batch_vec.max().item()) + 1)
        )
        graph_ids = (
            graph.graph_id.to(device)
            if hasattr(graph, "graph_id")
            else torch.arange(num_graphs, device=device)
        )
        counts = torch.bincount(batch_vec, minlength=num_graphs)
        out = torch.zeros(
            batch_vec.numel(), feature_dim, device=device, dtype=torch.float32
        )
        for graph_idx, gid in enumerate(graph_ids):
            mask = batch_vec == graph_idx
            count = int(counts[graph_idx].item())
            key = str(int(gid.item()))
            if key not in modules:
                modules[key] = TrainableFeatures(count, feature_dim)
            feats = modules[key](count).to(device)
            out[mask] = feats
        return out


def run_message_op(
    op: MessagePassing,
    x: tuple[torch.Tensor, torch.Tensor],
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run a PyG message passing op with optional edge attributes.

    Parameters
    ----------
    op : MessagePassing
        Message passing layer.
    x : tuple[torch.Tensor, torch.Tensor]
        Source and destination node features.
    edge_index : torch.Tensor
        Edge index tensor.
    edge_attr : Optional[torch.Tensor]
        Edge attributes, if supported by the op.

    Returns
    -------
    torch.Tensor
        Updated node features.
    """
    sig = inspect.signature(op.forward)
    if "edge_attr" in sig.parameters:
        return op(x, edge_index, edge_attr=edge_attr)
    if "edge_weight" in sig.parameters:
        edge_weight = (
            edge_attr[:, 0] if edge_attr is not None and edge_attr.dim() == 2 else None
        )
        return op(x, edge_index, edge_weight=edge_weight)
    return op(x, edge_index)


class SingleNodesetEncoder(nn.Module):
    """
    Embed source/destination features and apply message passing src -> dst.

    Parameters
    ----------
    embedder_src : nn.Module
        MLP mapping source node features into latent space.
    embedder_dst : nn.Module
        MLP mapping destination node features into latent space.
    message_op : MessagePassing
        PyG message passing layer (e.g., SAGEConv).
    post_linear : nn.Module, optional
        Additional linear applied to the dst output.

    Returns
    -------
    torch.Tensor
        Updated destination node embeddings.
    """

    def __init__(
        self,
        embedder_src: nn.Module,
        embedder_dst: nn.Module,
        message_op: MessagePassing,
        post_linear: Optional[nn.Module] = None,
    ):
        """Initialize the encoder.

        Parameters
        ----------
        embedder_src : nn.Module
            Source node embedder.
        embedder_dst : nn.Module
            Destination node embedder.
        message_op : MessagePassing
            Message passing layer.
        post_linear : Optional[nn.Module], optional
            Optional post-processing linear layer.

        Returns
        -------
        None
        """
        super().__init__()
        self.embedder_src = embedder_src
        self.embedder_dst = embedder_dst
        self.message_op = message_op
        self.post = post_linear
        self.activation = nn.GELU()

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode and propagate source to destination nodes.

        Parameters
        ----------
        x_src : torch.Tensor
            Source node features.
        x_dst : torch.Tensor
            Destination node features.
        edge_index : torch.Tensor
            Edge indices from source to destination.
        edge_attr : Optional[torch.Tensor], optional
            Edge attributes, if used by the message op.

        Returns
        -------
        torch.Tensor
            Updated destination embeddings.
        """
        x_src = self.activation(self.embedder_src(x_src))
        x_dst = self.activation(self.embedder_dst(x_dst))
        x_dst = self.activation(
            run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr)
        )
        if self.post is not None:
            x_dst = self.post(x_dst)
        return x_dst


class Processor(nn.Module):
    """
    Hidden-to-hidden message passing block.

    Parameters
    ----------
    message_op : MessagePassing
        PyG message passing layer for hidden nodes.
    hidden_dim : int
        Hidden dimension for MLPs following message passing.

    Returns
    -------
    torch.Tensor
        Updated hidden node embeddings.
    """

    def __init__(self, message_op: MessagePassing, hidden_dim: int):
        """Initialize the processor block.

        Parameters
        ----------
        message_op : MessagePassing
            Message passing layer for hidden nodes.
        hidden_dim : int
            Hidden feature dimension.

        Returns
        -------
        None
        """
        super().__init__()
        self.message_op = message_op
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run hidden-to-hidden message passing.

        Parameters
        ----------
        x_hidden : torch.Tensor
            Hidden node features.
        edge_index : torch.Tensor
            Edge indices for hidden-to-hidden edges.
        edge_attr : Optional[torch.Tensor], optional
            Edge attributes, if supported.

        Returns
        -------
        torch.Tensor
            Updated hidden features.
        """
        x_hidden = run_message_op(
            self.message_op, (x_hidden, x_hidden), edge_index, edge_attr
        )
        return self.mlp(x_hidden)


class SingleNodesetDecoder(nn.Module):
    """
    Decode hidden nodes to data nodes via message passing.

    Parameters
    ----------
    embedder_src : nn.Module
        MLP mapping hidden node features to latent space.
    embedder_dst : nn.Module
        MLP mapping data node features to latent space.
    message_op : MessagePassing
        PyG message passing layer for hidden-to-data edges.
    out_linear : nn.Module
        Projection from latent space to output channels.

    Returns
    -------
    torch.Tensor
        Decoded data node outputs.
    """

    def __init__(
        self,
        embedder_src: nn.Module,
        embedder_dst: nn.Module,
        message_op: MessagePassing,
        out_linear: nn.Module,
    ):
        """Initialize the decoder.

        Parameters
        ----------
        embedder_src : nn.Module
            Hidden node embedder.
        embedder_dst : nn.Module
            Data node embedder.
        message_op : MessagePassing
            Message passing layer.
        out_linear : nn.Module
            Output projection layer.

        Returns
        -------
        None
        """
        super().__init__()
        self.embedder_src = embedder_src
        self.embedder_dst = embedder_dst
        self.message_op = message_op
        self.out = out_linear
        self.activation = nn.GELU()

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode hidden features into data node predictions.

        Parameters
        ----------
        x_src : torch.Tensor
            Source (hidden) node features.
        x_dst : torch.Tensor
            Destination (data) node features.
        edge_index : torch.Tensor
            Edge indices for hidden-to-data edges.
        edge_attr : Optional[torch.Tensor], optional
            Edge attributes, if supported.

        Returns
        -------
        torch.Tensor
            Output predictions for data nodes.
        """
        x_src = self.activation(self.embedder_src(x_src))
        x_dst = self.activation(self.embedder_dst(x_dst))
        x_dst = self.activation(
            run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr)
        )
        return self.out(x_dst)


class EncodeProcessDecodeModel(nn.Module):
    """
    Encode -> process -> decode model for batched HeteroData graphs.
    """

    def __init__(
        self,
        encoder: SingleNodesetEncoder,
        processor: Processor,
        decoder: SingleNodesetDecoder,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        n_input_trainable_features: int = 0,
        n_hidden_trainable_features: int = 0,
        trainable_manager: Optional[TrainableFeatureManager] = None,
    ):
        """Initialize the encode-process-decode model.

        Parameters
        ----------
        encoder : SingleNodesetEncoder
            Encoder mapping data to hidden nodes.
        processor : Processor
            Hidden-to-hidden processor.
        decoder : SingleNodesetDecoder
            Decoder mapping hidden back to data.
        n_input_data_features : int, optional
            Input feature dimension on data nodes.
        n_output_data_features : int, optional
            Output feature dimension on data nodes.
        n_hidden_data_features : int, optional
            Feature dimension on hidden nodes.
        n_input_trainable_features : int, optional
            Trainable feature dimension on data nodes.
        n_hidden_trainable_features : int, optional
            Trainable feature dimension on hidden nodes.
        trainable_manager : Optional[TrainableFeatureManager], optional
            Optional shared trainable feature manager.

        Returns
        -------
        None
        """
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.n_input_trainable_features = n_input_trainable_features
        self.n_hidden_trainable_features = n_hidden_trainable_features
        self.trainable_manager = trainable_manager or TrainableFeatureManager(
            n_input_trainable_features, n_hidden_trainable_features
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """Run the encode-process-decode model on a graph.

        Parameters
        ----------
        graph : HeteroData
            Input graph with required node/edge types.

        Returns
        -------
        torch.Tensor
            Output predictions for data nodes.
        """
        required_nodes = {"data", "hidden"}
        required_edges = {
            ("data", "to", "hidden"),
            ("hidden", "to", "hidden"),
            ("hidden", "to", "data"),
        }
        assert required_nodes.issubset(
            set(graph.node_types)
        ), f"Graph missing nodes: {required_nodes - set(graph.node_types)}"
        assert required_edges.issubset(
            set(graph.edge_types)
        ), f"Graph missing edges: {required_edges - set(graph.edge_types)}"

        data_feats = graph["data"].x
        assert (
            data_feats.shape[1] == self.n_input_data_features
        ), f"Expected {self.n_input_data_features} data features, got {data_feats.shape[1]}"

        hidden_feats = graph["hidden"].x
        assert (
            hidden_feats.shape[1] == self.n_hidden_data_features
        ), f"Expected {self.n_hidden_data_features} hidden features, got {hidden_feats.shape[1]}"

        data_trainable = self.trainable_manager.build_features(graph, "data")
        hidden_trainable = self.trainable_manager.build_features(graph, "hidden")
        if data_trainable is not None:
            data_feats = torch.cat([data_feats, data_trainable], dim=-1)
        if hidden_trainable is not None:
            hidden_feats = torch.cat([hidden_feats, hidden_trainable], dim=-1)

        hidden_latent = self.encoder(
            x_src=data_feats,
            x_dst=hidden_feats,
            edge_index=graph["data", "to", "hidden"].edge_index,
            edge_attr=(
                graph["data", "to", "hidden"].edge_attr
                if "edge_attr" in graph["data", "to", "hidden"]
                else None
            ),
        )

        hidden_latent = self.processor(
            x_hidden=hidden_latent,
            edge_index=graph["hidden", "to", "hidden"].edge_index,
            edge_attr=(
                graph["hidden", "to", "hidden"].edge_attr
                if "edge_attr" in graph["hidden", "to", "hidden"]
                else None
            ),
        )

        hidden_with_attrs = torch.cat([hidden_latent, graph["hidden"].x], dim=-1)
        if hidden_trainable is not None:
            hidden_with_attrs = torch.cat([hidden_with_attrs, hidden_trainable], dim=-1)

        data_out = self.decoder(
            x_src=hidden_with_attrs,
            x_dst=data_feats,
            edge_index=graph["hidden", "to", "data"].edge_index,
            edge_attr=(
                graph["hidden", "to", "data"].edge_attr
                if "edge_attr" in graph["hidden", "to", "data"]
                else None
            ),
        )

        return data_out

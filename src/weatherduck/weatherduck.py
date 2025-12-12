import inspect
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import fiddle
import fiddle.experimental
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import MessagePassing, SAGEConv


# ============================================================
# Graph helpers
# ============================================================
def build_dummy_weather_graph(
    num_data_nodes: int = 64,
    num_hidden_nodes: int = 32,
    edge_attr_dim: int = 2,
    n_data_node_features: int = 0,
    n_hidden_node_features: int = 0,
) -> HeteroData:
    """
    Build a minimal heterogeneous graph with the expected topology:
    data -> hidden, hidden -> hidden, hidden -> data.

    Parameters
    ----------
    num_data_nodes : int, default 64
        Number of data nodes.
    num_hidden_nodes : int, default 32
        Number of hidden nodes.
    edge_attr_dim : int, default 2
        Dimension of edge attributes.
    n_data_node_features : int, default 0
        Feature length on data nodes (non-trainable).
    n_hidden_node_features : int, default 0
        Feature length on hidden nodes (non-trainable).

    Returns
    -------
    HeteroData
        Graph with node/edge attributes populated with random tensors.
    """
    graph = HeteroData()
    graph["data"].x = torch.randn(num_data_nodes, n_data_node_features)
    graph["hidden"].x = torch.randn(num_hidden_nodes, n_hidden_node_features)

    def dense_edges(n_src: int, n_dst: int, fanout: int) -> torch.Tensor:
        src = torch.arange(n_src).repeat_interleave(fanout)
        dst_choices = torch.randint(0, n_dst, (n_src * fanout,))
        return torch.stack([src, dst_choices], dim=0)

    graph["data", "to", "hidden"].edge_index = dense_edges(num_data_nodes, num_hidden_nodes, fanout=4)
    graph["hidden", "to", "hidden"].edge_index = dense_edges(num_hidden_nodes, num_hidden_nodes, fanout=6)
    graph["hidden", "to", "data"].edge_index = dense_edges(num_hidden_nodes, num_data_nodes, fanout=4)

    for key in [
        ("data", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "data"),
    ]:
        num_edges = graph[key].edge_index.shape[1]
        graph[key].edge_attr = torch.randn(num_edges, edge_attr_dim)

    return graph


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


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
        super().__init__()
        self.num_nodes = num_nodes
        self.n_features = n_features
        param = nn.Parameter(torch.zeros(num_nodes, n_features))
        nn.init.constant_(param, 0.0)
        self.register_parameter("trainable", param)

    def forward(self, current_num_nodes: int) -> torch.Tensor:
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


def run_message_op(
    op: MessagePassing,
    x: tuple[torch.Tensor, torch.Tensor],
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Call a MessagePassing op while gracefully ignoring edge_attr if unsupported.
    """
    sig = inspect.signature(op.forward)
    if "edge_attr" in sig.parameters:
        return op(x, edge_index, edge_attr=edge_attr)
    if "edge_weight" in sig.parameters:
        edge_weight = edge_attr[:, 0] if edge_attr is not None and edge_attr.dim() == 2 else None
        return op(x, edge_index, edge_weight=edge_weight)
    return op(x, edge_index)


# ============================================================
# Core model blocks
# ============================================================
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
        Linear layer applied after message passing.

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
        super().__init__()
        self.embedder_src = embedder_src
        self.embedder_dst = embedder_dst
        self.message_op = message_op
        self.post = post_linear or nn.Identity()
        self.activation = nn.GELU()

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_src = self.activation(self.embedder_src(x_src))
        x_dst = self.activation(self.embedder_dst(x_dst))
        x_dst = run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr)
        x_dst = self.activation(self.post(x_dst))
        return x_dst


class Processor(nn.Module):
    """
    Single message-passing update on hidden->hidden edges with residual + norm.

    Parameters
    ----------
    message_op : MessagePassing
        PyG message passing layer for hidden-to-hidden edges.
    hidden_dim : int
        Feature dimension of hidden nodes.

    Returns
    -------
    torch.Tensor
        Updated hidden node embeddings.
    """

    def __init__(self, message_op: MessagePassing, hidden_dim: int):
        super().__init__()
        self.message_op = message_op
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x_hidden
        x_hidden = self.activation(run_message_op(self.message_op, (x_hidden, x_hidden), edge_index, edge_attr))
        x_hidden = self.norm(x_hidden + residual)
        return x_hidden


class SingleNodesetDecoder(nn.Module):
    """
    Embed hidden/data nodes, pass messages hidden -> data, project to outputs.

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
        x_src = self.activation(self.embedder_src(x_src))
        x_dst = self.activation(self.embedder_dst(x_dst))
        x_dst = self.activation(run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr))
        return self.out(x_dst)


class EncodeProcessDecodeModel(nn.Module):
    """
    Encode -> process -> decode model for batched HeteroData graphs.

    Parameters
    ----------
    encoder : SingleNodesetEncoder
        Module mapping data nodes to hidden nodes.
    processor : Processor
        Module updating hidden nodes via message passing.
    decoder : SingleNodesetDecoder
        Module mapping hidden nodes back to data nodes.
    n_input_data_features : int, default 8
        Number of dataset-provided data node features.
    n_output_data_features : int, default 8
        Number of output channels on data nodes.
    n_hidden_data_features : int, default 0
        Number of dataset-provided hidden node features.
    n_input_trainable_features : int, default 0
        Learnable feature length per data node.
    n_trainable_hidden_features : int, default 0
        Learnable feature length per hidden node.

    Returns
    -------
        torch.Tensor
        Output tensor for all data nodes in the (possibly batched) graph.
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
        n_trainable_hidden_features: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.n_input_trainable_features = n_input_trainable_features
        self.n_trainable_hidden_features = n_trainable_hidden_features
        self.trainable_data_modules = nn.ModuleDict()
        self.trainable_hidden_modules = nn.ModuleDict()

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward pass on a (possibly batched) HeteroData graph.

        Parameters
        ----------
        graph : HeteroData
            Must contain:
            - node types: {'data', 'hidden'}
                * graph['data'].x: [N_data, n_input_data_features]
                    Initial state for forward prediction. E.g. two timesteps
                    of initial state features, forcing features and static
                    features all concatenated into "data features".
                * graph['hidden'].x: [N_hidden, n_hidden_data_features]
                    Hidden-node features. E.g. often positional/metadata. Can
                    be zero-dim if no hidden attributes are provided.
                * graph.graph_id: [num_graphs] unique identifier per graph in the batch (used to scope trainable features).
            - edge types: {('data','to','hidden'), ('hidden','to','hidden'), ('hidden','to','data')}
                * graph[('data','to','hidden')].edge_index: [2, E_dh]
                    adjecency list of edges from data -> hidden nodes, i.e. the encoder step.
                * graph[('data','to','hidden')].edge_attr: [E_dh, edge_attr_dim] optional
                    edge attributes for encoder edges, e.g. relative position of data->hidden nodes.
                * graph[('hidden','to','hidden')].edge_index: [2, E_hh]
                    adjecency list of edges from hidden -> hidden nodes, i.e. the processor step.
                * graph[('hidden','to','hidden')].edge_attr: [E_hh, edge_attr_dim] optional
                    edge attributes for processor edges, e.g. relative position of hidden->hidden nodes.
                * graph[('hidden','to','data')].edge_index: [2, E_hd]
                    adjecency list of edges from hidden -> data nodes, i.e. the decoder step.
                * graph[('hidden','to','data')].edge_attr: [E_hd, edge_attr_dim] optional
                    edge attributes for decoder edges, e.g. relative position of hidden->data nodes.

            Batched graphs are treated as disconnected; trainable features are
            repeated per-graph if batch multiples of the template node counts.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [total_data_nodes, n_output_data_features].
        """
        required_nodes = {"data", "hidden"}
        required_edges = {
            ("data", "to", "hidden"),
            ("hidden", "to", "hidden"),
            ("hidden", "to", "data"),
        }
        assert required_nodes.issubset(set(graph.node_types)), f"Graph missing nodes: {required_nodes - set(graph.node_types)}"
        assert required_edges.issubset(set(graph.edge_types)), f"Graph missing edges: {required_edges - set(graph.edge_types)}"

        data_feats = graph["data"].x

        hidden_feats = graph["hidden"].x
        # ensure hidden_feats includes hidden data features
        if hidden_feats.shape[1] == 0 and self.n_hidden_data_features > 0:
            hidden_feats = hidden_feats.new_zeros(graph["hidden"].num_nodes, self.n_hidden_data_features)

        # build per-node trainable features using graph ids
        num_graphs = graph.num_graphs if hasattr(graph, "num_graphs") else 1
        graph_ids = (
            graph.graph_id.to(data_feats.device)
            if hasattr(graph, "graph_id")
            else torch.arange(num_graphs, device=data_feats.device)
        )
        data_batch = graph["data"].batch if "batch" in graph["data"] else torch.zeros(data_feats.size(0), dtype=torch.long)
        hidden_batch = graph["hidden"].batch if "batch" in graph["hidden"] else torch.zeros(hidden_feats.size(0), dtype=torch.long)
        data_counts = torch.bincount(data_batch, minlength=num_graphs)
        hidden_counts = torch.bincount(hidden_batch, minlength=num_graphs)
        data_trainable = self._build_trainable_features(
            self.trainable_data_modules,
            graph_ids,
            data_batch.to(data_feats.device),
            self.n_input_trainable_features,
            data_counts,
        )
        hidden_trainable = self._build_trainable_features(
            self.trainable_hidden_modules,
            graph_ids,
            hidden_batch.to(hidden_feats.device),
            self.n_trainable_hidden_features,
            hidden_counts,
        )
        if data_trainable is not None:
            data_feats = torch.cat([data_feats, data_trainable], dim=-1)
        if hidden_trainable is not None:
            hidden_feats = torch.cat([hidden_feats, hidden_trainable], dim=-1)

        hidden_latent = self.encoder(
            x_src=data_feats,
            x_dst=hidden_feats,
            edge_index=graph["data", "to", "hidden"].edge_index,
            edge_attr=graph["data", "to", "hidden"].edge_attr if "edge_attr" in graph["data", "to", "hidden"] else None,
        )

        hidden_latent = self.processor(
            x_hidden=hidden_latent,
            edge_index=graph["hidden", "to", "hidden"].edge_index,
            edge_attr=graph["hidden", "to", "hidden"].edge_attr if "edge_attr" in graph["hidden", "to", "hidden"] else None,
        )

        hidden_with_attrs = torch.cat([hidden_latent, graph["hidden"].x], dim=-1)
        if graph["hidden"].x.shape[1] == 0 and self.n_hidden_data_features > 0:
            hidden_with_attrs = torch.cat(
                [hidden_with_attrs, hidden_with_attrs.new_zeros(graph["hidden"].num_nodes, self.n_hidden_data_features)],
                dim=-1,
            )
        if hidden_trainable is not None:
            hidden_with_attrs = torch.cat([hidden_with_attrs, hidden_trainable], dim=-1)

        data_out = self.decoder(
            x_src=hidden_with_attrs,
            x_dst=data_feats,
            edge_index=graph["hidden", "to", "data"].edge_index,
            edge_attr=graph["hidden", "to", "data"].edge_attr if "edge_attr" in graph["hidden", "to", "data"] else None,
        )

        return data_out

    @staticmethod
    def _build_trainable_features(
        module_dict: nn.ModuleDict,
        graph_ids: torch.Tensor,
        batch_vec: torch.Tensor,
        feature_dim: int,
        num_nodes_per_graph: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Construct per-node trainable features for a batched graph using per-graph ids.

        Parameters
        ----------
        module_dict : nn.ModuleDict
            Storage for TrainableFeatures modules keyed by graph id.
        graph_ids : torch.Tensor
            Unique identifier per graph in the batch, shape [num_graphs].
        batch_vec : torch.Tensor
            Batch vector for nodes (data or hidden), shape [num_nodes].
        feature_dim : int
            Feature dimension to allocate per node.
        num_nodes_per_graph : torch.Tensor
            Node counts per graph, shape [num_graphs].

        Returns
        -------
        torch.Tensor | None
            Trainable feature tensor aligned with nodes or None if feature_dim == 0.
        """
        if feature_dim == 0:
            return None
        device = batch_vec.device
        out = torch.zeros(batch_vec.numel(), feature_dim, device=device, dtype=torch.float32)
        for graph_idx, gid in enumerate(graph_ids):
            mask = batch_vec == graph_idx
            count = int(num_nodes_per_graph[graph_idx].item())
            key = str(int(gid.item()))
            if key not in module_dict:
                module_dict[key] = TrainableFeatures(count, feature_dim)
            feats = module_dict[key](count).to(device)
            out[mask] = feats
        return out


# ============================================================
# LightningModule wrapper
# ============================================================
class LitWeatherDuck(pl.LightningModule):
    """
    Lightning wrapper around the WeatherEncProcDec model.

    Parameters
    ----------
    model : EncodeProcessDecodeModel
        Core GNN model to train/evaluate.
    lr : float, default 1e-3
        Learning rate for the Adam optimizer.

    Returns
    -------
    torch.Tensor
        Model predictions on the provided HeteroData batch.
    """

    def __init__(self, model: EncodeProcessDecodeModel, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x: HeteroData) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, batch_size=1)
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=1)

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ============================================================
# Dummy data module (placeholder until real weather data is wired)
# ============================================================
class DummyWeatherDataset(Dataset):
    """
    Dummy dataset producing random HeteroData samples for quick execution.

    Parameters
    ----------
    num_samples : int
        Number of graphs to generate.
    num_data_nodes : int | dict[int, int]
        Number of data nodes per graph, or mapping graph_id -> node count.
    n_input_data_features : int
        Number of data input features (excluding trainable features).
    n_output_data_features : int
        Number of target channels.
    n_unique_graphs : int, default 1
        Number of unique graphs to cycle through when sampling.

    Returns
    -------
    HeteroData
        Random graph with features and targets on data nodes.
    """

    def __init__(
        self,
        num_samples: int,
        num_data_nodes: int | dict[int, int],
        n_input_data_features: int,
        n_output_data_features: int,
        n_hidden_data_features: int,
        n_unique_graphs: int = 1,
    ):
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.n_unique_graphs = n_unique_graphs
        self.graphs: list[HeteroData] = []
        for gid in range(n_unique_graphs):
            if isinstance(self.num_data_nodes, dict):
                num_nodes = self.num_data_nodes[gid]
            else:
                num_nodes = self.num_data_nodes
            g = build_dummy_weather_graph(
                num_data_nodes=num_nodes,
                num_hidden_nodes=max(1, num_nodes // 2),
                edge_attr_dim=2,
                n_data_node_features=0,
                n_hidden_node_features=self.n_hidden_data_features,
            )
            g.graph_id = torch.tensor([gid], dtype=torch.long)
            self.graphs.append(g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        graph = self.graphs[idx % self.n_unique_graphs].clone()
        # pick node count for this graph id
        if isinstance(self.num_data_nodes, dict):
            gid = int(graph.graph_id.item())
            num_data_nodes = self.num_data_nodes.get(gid)
            assert num_data_nodes is not None, f"num_data_nodes missing entry for graph id {gid}"
        else:
            num_data_nodes = self.num_data_nodes

        graph["data"].x = torch.randn(num_data_nodes, self.n_input_data_features)
        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(graph["hidden"].num_nodes, self.n_hidden_data_features)
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        graph["data"].y = torch.randn(num_data_nodes, self.n_output_data_features)
        return graph

    def collate_fn(self, graphs: list[HeteroData]) -> Batch:
        # Ensure each graph already has a unique graph_id; Batch will merge into a tensor
        return Batch.from_data_list(graphs)


@fiddle.experimental.auto_config.auto_config
def build_encode_process_decode_model(
    *,
    n_input_data_features: int,
    n_output_data_features: int,
    n_hidden_data_features: int,
    n_input_trainable_features: int,
    n_trainable_hidden_features: int,
    hidden_dim: int,
) -> EncodeProcessDecodeModel:
    """
    Factory to build an EncodeProcessDecodeModel with SAGEConv components.

    Parameters
    ----------
    n_input_data_features : int
        Dataset-provided data node features.
    n_output_data_features : int
        Decoder output channels.
    n_hidden_data_features : int
        Dataset-provided hidden node features.
    n_input_trainable_features : int
        Trainable feature length per data node.
    n_trainable_hidden_features : int
        Trainable feature length per hidden node.
    hidden_dim : int
        Latent dimension for encoder/processor/decoder.

    Returns
    -------
    EncodeProcessDecodeModel
    """
    encoder = SingleNodesetEncoder(
        embedder_src=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        embedder_dst=make_mlp(n_hidden_data_features + n_trainable_hidden_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=nn.Linear(hidden_dim, hidden_dim),
    )
    processor = Processor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = SingleNodesetDecoder(
        embedder_src=make_mlp(
            hidden_dim + n_hidden_data_features + n_trainable_hidden_features, hidden_dim, hidden_dim
        ),
        embedder_dst=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=nn.Linear(hidden_dim, n_output_data_features),
    )

    return EncodeProcessDecodeModel(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_trainable_hidden_features=n_trainable_hidden_features,
    )


class WeatherDuckDataModule(pl.LightningDataModule):
    """
    LightningDataModule providing dummy weather graphs via PyG DataLoader.

    Parameters
    ----------
    num_samples : int, default 128
        Number of training samples.
    num_data_nodes : int, default 64
        Number of data nodes per graph, or a dict mapping graph_id -> node count (len must equal n_unique_graphs).
    n_input_data_features : int, default 8
        Number of data input channels (excluding trainable features).
    n_output_data_features : int, default 8
        Number of target channels.
    n_hidden_data_features : int, default 0
        Number of dataset-provided hidden node features.
    batch_size : int, default 4
        Batch size for PyG DataLoader.

    Returns
    -------
    DataLoader
        PyG loaders for train/val/test splits.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.batch_size = batch_size
        self.n_unique_graphs = n_unique_graphs

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = DummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.val_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.test_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )

    def train_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn)

    def test_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn)


# ============================================================
# Experiment dataclass
# ============================================================
@dataclass
class Experiment:
    """
    Container bundling model, data module, and trainer.

    Parameters
    ----------
    model : LitWeatherDuck
        Lightning model to train/test.
    data : WeatherDuckDataModule
        Data module providing graph batches.
    trainer : pl.Trainer
        PyTorch Lightning trainer.

    Returns
    -------
    None
    """

    model: LitWeatherDuck
    data: WeatherDuckDataModule
    trainer: pl.Trainer

    def run(self) -> None:
        self.trainer.fit(self.model, datamodule=self.data)
        self.trainer.test(self.model, datamodule=self.data)


# ============================================================
# Fiddle auto_config wiring -> Config[Experiment]
# ============================================================
@fiddle.experimental.auto_config.auto_config
def experiment_factory() -> Experiment:
    """
    Build a Fiddle config graph that mirrors the Hydra GNN setup
    (encode -> process -> decode) but runs with dummy graph/data.
    """
    n_input_data_features = 8
    n_output_data_features = 8
    hidden_dim = 128
    n_hidden_data_features = 4
    n_input_trainable_features = 2
    n_trainable_hidden_features = 3
    core_model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_trainable_hidden_features=n_trainable_hidden_features,
        hidden_dim=hidden_dim,
    )

    lit_module = LitWeatherDuck(
        model=core_model,
        lr=1e-3,
    )

    data = WeatherDuckDataModule(
        num_samples=256,
        num_data_nodes=64,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=4,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
    )

    return Experiment(
        model=lit_module,
        data=data,
        trainer=trainer,
    )

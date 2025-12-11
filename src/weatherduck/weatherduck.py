import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import fiddle
import fiddle.experimental
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
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
        assert (
            current_num_nodes == self.num_nodes
        ), f"Trainable features expect {self.num_nodes} nodes, got {current_num_nodes}"
        return self.trainable


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
    embed_src : nn.Module
        MLP mapping source node features into latent space.
    embed_dst : nn.Module
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
        embed_src: nn.Module,
        embed_dst: nn.Module,
        message_op: MessagePassing,
        post_linear: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_src = embed_src
        self.embed_dst = embed_dst
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
        x_src = self.activation(self.embed_src(x_src))
        x_dst = self.activation(self.embed_dst(x_dst))
        x_dst = run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr)
        x_dst = self.activation(self.post(x_dst))
        return x_dst


class WeatherProcessor(nn.Module):
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
    embed_src : nn.Module
        MLP mapping hidden node features to latent space.
    embed_dst : nn.Module
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
        embed_src: nn.Module,
        embed_dst: nn.Module,
        message_op: MessagePassing,
        out_linear: nn.Module,
    ):
        super().__init__()
        self.embed_src = embed_src
        self.embed_dst = embed_dst
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
        x_src = self.activation(self.embed_src(x_src))
        x_dst = self.activation(self.embed_dst(x_dst))
        x_dst = self.activation(run_message_op(self.message_op, (x_src, x_dst), edge_index, edge_attr))
        return self.out(x_dst)


class WeatherEncProcDec(nn.Module):
    """
    Encode -> process -> decode model for batched HeteroData graphs.

    Parameters
    ----------
    graph : HeteroData
        Template graph used for static attributes and topology validation.
    encoder : SingleNodesetEncoder
        Module mapping data nodes to hidden nodes.
    processor : WeatherProcessor
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
        graph: HeteroData,
        encoder: SingleNodesetEncoder,
        processor: WeatherProcessor,
        decoder: SingleNodesetDecoder,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        n_input_trainable_features: int = 0,
        n_trainable_hidden_features: int = 0,
    ):
        super().__init__()
        self.graph = graph
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.n_input_trainable_features = n_input_trainable_features
        self.n_trainable_hidden_features = n_trainable_hidden_features
        self.trainable_data_feats = (
            TrainableFeatures(graph["data"].num_nodes, n_input_trainable_features)
            if n_input_trainable_features > 0
            else None
        )
        self.trainable_hidden_feats = (
            TrainableFeatures(graph["hidden"].num_nodes, n_trainable_hidden_features)
            if n_trainable_hidden_features > 0
            else None
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Forward pass on a (possibly batched) HeteroData graph.

        Parameters
        ----------
        graph : HeteroData
            Must contain node types {'data','hidden'} and edges
            ('data','to','hidden'), ('hidden','to','hidden'),
            ('hidden','to','data').

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
        if self.trainable_data_feats is not None:
            data_feats = torch.cat(
                [data_feats, self.trainable_data_feats(graph["data"].num_nodes).to(data_feats.device)],
                dim=-1,
            )

        hidden_feats = graph["hidden"].x
        # ensure hidden_feats includes hidden data features
        if hidden_feats.shape[1] == 0 and self.n_hidden_data_features > 0:
            hidden_feats = hidden_feats.new_zeros(graph["hidden"].num_nodes, self.n_hidden_data_features)
        if self.trainable_hidden_feats is not None:
            hidden_feats = torch.cat(
                [hidden_feats, self.trainable_hidden_feats(graph["hidden"].num_nodes).to(hidden_feats.device)],
                dim=-1,
            )

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

        hidden_with_attrs = hidden_latent
        hidden_with_attrs = torch.cat([hidden_with_attrs, graph["hidden"].x], dim=-1)
        if graph["hidden"].x.shape[1] == 0 and self.n_hidden_data_features > 0:
            hidden_with_attrs = torch.cat(
                [hidden_with_attrs, hidden_with_attrs.new_zeros(graph["hidden"].num_nodes, self.n_hidden_data_features)],
                dim=-1,
            )
        if self.trainable_hidden_feats is not None:
            hidden_with_attrs = torch.cat(
                [hidden_with_attrs, self.trainable_hidden_feats(graph["hidden"].num_nodes).to(hidden_with_attrs.device)],
                dim=-1,
            )

        data_out = self.decoder(
            x_src=hidden_with_attrs,
            x_dst=data_feats,
            edge_index=graph["hidden", "to", "data"].edge_index,
            edge_attr=graph["hidden", "to", "data"].edge_attr if "edge_attr" in graph["hidden", "to", "data"] else None,
        )

        return data_out


# ============================================================
# LightningModule wrapper
# ============================================================
class LitWeatherDuck(pl.LightningModule):
    """
    Lightning wrapper around the WeatherEncProcDec model.

    Parameters
    ----------
    model : WeatherEncProcDec
        Core GNN model to train/evaluate.
    lr : float, default 1e-3
        Learning rate for the Adam optimizer.

    Returns
    -------
    torch.Tensor
        Model predictions on the provided HeteroData batch.
    """

    def __init__(self, model: WeatherEncProcDec, lr: float = 1e-3):
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)

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
    num_data_nodes : int
        Number of data nodes per graph.
    n_input_data_features : int
        Number of data input features (excluding trainable features).
    n_output_data_features : int
        Number of target channels.

    Returns
    -------
    HeteroData
        Random graph with features and targets on data nodes.
    """

    def __init__(
        self,
        num_samples: int,
        num_data_nodes: int,
        n_input_data_features: int,
        n_output_data_features: int,
        n_hidden_data_features: int,
    ):
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        graph = build_dummy_weather_graph(
            num_data_nodes=self.num_data_nodes,
            num_hidden_nodes=max(1, self.num_data_nodes // 2),
            edge_attr_dim=2,
            n_data_node_features=0,
            n_hidden_node_features=self.n_hidden_data_features,
        )
        graph["data"].x = torch.randn(self.num_data_nodes, self.n_input_data_features)
        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(graph["hidden"].num_nodes, self.n_hidden_data_features)
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        graph["data"].y = torch.randn(self.num_data_nodes, self.n_output_data_features)
        return graph


class WeatherDuckDataModule(pl.LightningDataModule):
    """
    LightningDataModule providing dummy weather graphs via PyG DataLoader.

    Parameters
    ----------
    num_samples : int, default 128
        Number of training samples.
    num_data_nodes : int, default 64
        Number of data nodes per graph.
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
        num_data_nodes: int = 64,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        batch_size: int = 4,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = DummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
        )
        self.val_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
        )
        self.test_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
        )

    def train_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.test_ds, batch_size=self.batch_size)


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

    graph = build_dummy_weather_graph(
        num_data_nodes=64,
        num_hidden_nodes=32,
        edge_attr_dim=2,
        n_data_node_features=0,
        n_hidden_node_features=n_hidden_data_features,
    )

    encoder = SingleNodesetEncoder(
        embed_src=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        embed_dst=make_mlp(n_trainable_hidden_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=nn.Linear(hidden_dim, hidden_dim),
    )
    processor = WeatherProcessor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = SingleNodesetDecoder(
        embed_src=make_mlp(hidden_dim + n_trainable_hidden_features, hidden_dim, hidden_dim),
        embed_dst=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=nn.Linear(hidden_dim, n_output_data_features),
    )

    core_model = WeatherEncProcDec(
        graph=graph,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_trainable_hidden_features=n_trainable_hidden_features,
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

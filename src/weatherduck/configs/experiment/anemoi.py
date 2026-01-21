import fiddle.experimental.auto_config
import pytorch_lightning as pl
from anemoi.graphs.create import GraphBuilder
from anemoi.graphs.edges import CutOffEdges, KNNEdges, MultiScaleEdges
from anemoi.graphs.edges.attributes import EdgeDirection, EdgeLength
from anemoi.graphs.nodes import AnemoiDatasetNodes, LimitedAreaTriNodes
from anemoi.graphs.nodes.attributes import CutOutMask

from ...ar_forecaster import AutoRegressiveForecaster
from ...data.anemoi import AnemoiNativeGridDataModule
from ...lightning import WeatherDuckModule
from ..model.base import build_encode_process_decode_model
from .base import Experiment

__all__ = ["create_lam_graph_builder", "experiment_factory"]


@fiddle.experimental.auto_config.auto_config
def create_lam_graph_builder(
    dataset_config: dict,
    *,
    resolution: int,
    margin_radius_km: int,
    cutoff_factor: float,
    num_nearest_neighbours: int,
) -> GraphBuilder:
    """Build a GraphBuilder for a limited-area Anemoi graph.

    Parameters
    ----------
    dataset_config : dict
        Dataset configuration passed to Anemoi graph nodes.
    resolution : int
        Limited-area triangle mesh resolution.
    margin_radius_km : int
        Margin radius used when building the hidden mesh.
    cutoff_factor : float
        Cutoff factor for connecting data to hidden nodes.
    num_nearest_neighbours : int
        Number of neighbours for hidden-to-data KNN edges.

    Returns
    -------
    GraphBuilder
        Configured graph builder.
    """
    edge_attrs = [
        EdgeLength(name="edge_length", norm="unit-std"),
        EdgeDirection(name="edge_dirs", norm="unit-std"),
    ]
    nodes = [
        AnemoiDatasetNodes(
            name="data",
            dataset=dataset_config,
            attributes=[CutOutMask(name="cutout_mask")],
        ),
        LimitedAreaTriNodes(
            name="hidden",
            resolution=resolution,
            reference_node_name="data",
            mask_attr_name="cutout_mask",
            margin_radius_km=margin_radius_km,
        ),
    ]
    edges = [
        CutOffEdges(
            source_name="data",
            target_name="hidden",
            cutoff_factor=cutoff_factor,
            attributes=edge_attrs,
        ),
        MultiScaleEdges(
            source_name="hidden",
            target_name="hidden",
            x_hops=1,
            scale_resolutions=resolution,
            attributes=edge_attrs,
        ),
        KNNEdges(
            source_name="hidden",
            target_name="data",
            num_nearest_neighbours=num_nearest_neighbours,
            attributes=edge_attrs,
        ),
    ]
    return GraphBuilder(nodes=nodes, edges=edges)


@fiddle.experimental.auto_config.auto_config
def experiment_factory(
    *,
    dataset_config: dict,
    split_config: dict,
    n_input_data_features: int,
    n_hidden_data_features: int = 2,
    data_frequency: str = "6h",
    data_timestep: str = "6h",
    multistep_input: int = 1,
    rollout: int = 1,
    validation_rollout: int = 1,
    resolution: int = 2,
    margin_radius_km: int = 10,
    cutoff_factor: float = 0.6,
    num_nearest_neighbours: int = 3,
) -> Experiment:
    """Build an Experiment for Anemoi-backed autoregressive training.

    Parameters
    ----------
    dataset_config : dict
        Dataset configuration for Anemoi datasets.
    split_config : dict
        Split configuration used for training/validation/testing.
    n_input_data_features : int
        Number of input data variables.
    n_hidden_data_features : int, optional
        Number of hidden data features, by default 2.
    data_frequency : str, optional
        Data sampling frequency, by default "6h".
    data_timestep : str, optional
        Dataset timestep, by default "6h".
    multistep_input : int, optional
        Number of input steps per sample, by default 1.
    rollout : int, optional
        Autoregressive rollout steps, by default 1.
    validation_rollout : int, optional
        Validation rollout steps, by default 1.
    resolution : int, optional
        Hidden mesh resolution, by default 2.
    margin_radius_km : int, optional
        Margin radius for the hidden mesh, by default 10.
    cutoff_factor : float, optional
        Cutoff factor for data-to-hidden edges, by default 0.6.
    num_nearest_neighbours : int, optional
        Nearest neighbours for hidden-to-data edges, by default 3.

    Returns
    -------
    Experiment
        Experiment containing model, data, and trainer.
    """
    graph_builder = create_lam_graph_builder(
        dataset_config,
        resolution=resolution,
        margin_radius_km=margin_radius_km,
        cutoff_factor=cutoff_factor,
        num_nearest_neighbours=num_nearest_neighbours,
    )

    step_model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_input_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=0,
        n_hidden_trainable_features=0,
        hidden_dim=64,
    )
    ar_model = AutoRegressiveForecaster(step_predictor=step_model)
    model = WeatherDuckModule(model=ar_model, lr=1e-3)

    data = AnemoiNativeGridDataModule(
        graph_builder=graph_builder,
        training=split_config,
        validation=split_config,
        test=split_config,
        data_frequency=data_frequency,
        data_timestep=data_timestep,
        multistep_input=multistep_input,
        rollout=rollout,
        validation_rollout=validation_rollout,
        batch_size={"training": 1, "validation": 1, "test": 1},
        num_workers={"training": 0, "validation": 0, "test": 0},
        pin_memory=False,
        ar_steps=rollout,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )

    return Experiment(pl_module=model, data=data, trainer=trainer)

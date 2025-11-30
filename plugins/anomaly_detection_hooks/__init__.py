from .anomaly_detection_plugins.devnet_plugin import DevNetPlugin  # noqa
from .anomaly_detection_plugins.vitae_plugin import ViTAEPlugin
from .anomaly_detection_plugin_manager import AnomalyInferenceInput  # noqa
from .anomaly_detection_plugin_manager import AnomalyInferenceOutput  # noqa
from .anomaly_detection_plugin_manager import AnomalyInferencePlugin  # noqa
from .anomaly_detection_plugin_manager import AnomalyInferencePluginConfig  # noqa
from .anomaly_detection_plugin_manager import AnomalyInferencePluginManager  # noqa
from .anomaly_detection_plugin_manager import default_anomaly_plugin_registry  # noqa
from .anomaly_detection_plugin_manager import (
    register_anomaly_inference_plugin,  # noqa
    remove_from_anomaly_inference_plugin_registry,  # noqa
    anomaly_inference_plugin_registry_mapping,  # noqa
)  # noqa; noqa

ANOMALY_PLUGIN_CLASSES = {
    "devnet": DevNetPlugin,
    "vitae": ViTAEPlugin,
}
